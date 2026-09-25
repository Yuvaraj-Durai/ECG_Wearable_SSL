"""Load-aware job scheduler for both GPUs.

Every job has a GPU-load weight; a ready job (dependencies present) is started on the GPU
with the lowest load that has enough free memory. Load = weights of this scheduler's jobs
+ jobs of this project started elsewhere + a fixed charge for other users' processes.
Finished jobs (result file present) are skipped, so the scheduler can be restarted.

  .venv/bin/python scripts/scheduler.py
"""
import os
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
PY = ".venv/bin/python"
CAP = float(os.environ.get("CAP", 5))  # load units per GPU
FOREIGN = 2.0  # load charged for another user's process on a GPU
SEEDS = [0, 1, 2]
PT = "checkpoints/pretrain/moco.pt"
PT_DONE = "checkpoints/pretrain/moco.DONE"
NOISY = "data/cache/chapman_test_noisy_X.npy"
CPSC = "data/cache/cpsc2018_X.npy"


class Job:
    def __init__(self, name, cmd, done, deps=(), weight=1.0, mem=10000):
        self.name, self.cmd, self.done, self.deps, self.weight, self.mem = name, cmd, done, list(deps), weight, mem
        self.log = f"logs/{name.replace('/', '_')}.log"

    def ready(self):
        return all(os.path.exists(d) for d in self.deps)


def run(tag, args, deps=(), dataset="chapman", weight=1.0):
    return Job(f"{dataset}/{tag}", f"{PY} -m src.train --dataset {dataset} --tag {tag} {args}",
               f"results/runs/{dataset}/{tag}.json", [NOISY, *deps], weight)


def jobs():
    J = []
    for s in SEEDS:
        # Fig. 2a ablation (+ single-scale MSDNN to isolate the multiscale layer)
        J += [run(f"dnn_s{s}", f"--model dnn --seed {s}", weight=1.5),
              run(f"msdnn_s{s}", f"--seed {s}"),
              run(f"ssdnn_s{s}", f"--model ssdnn --seed {s}"),
              run(f"msdnn_aug_s{s}", f"--aug all --seed {s}"),
              run(f"msdnn_pw_s{s}", f"--init {PT} --seed {s}", [PT_DONE]),
              run(f"msdnn_pw_aug_s{s}", f"--init {PT} --aug all --seed {s}", [PT_DONE])]
        # Fig. 2c: each augmentation alone
        J += [run(f"msdnn_aug-{op}_s{s}", f"--aug {op} --seed {s}") for op in ["freq", "crop", "cycle", "channel"]]
        # Fig. 2b: training-set size, random vs pretrained initialisation
        for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if s == 0 else []:  # one seed, as in the paper
            J.append(run(f"msdnn_frac{f}_s{s}", f"--frac {f} --seed {s}", weight=0.5 + 0.5 * f))
            J.append(run(f"msdnn_pw_frac{f}_s{s}", f"--init {PT} --frac {f} --seed {s}", [PT_DONE], weight=0.5 + 0.5 * f))
        # Fig. 2e: 1-3 lead devices simulated from the 12 leads
        for name, L in [("I", "I"), ("holter", "II,V1,V5"), ("frank", "I,aVF,V2")]:
            J.append(run(f"msdnn_pw_aug_lead-{name}_s{s}", f"--init {PT} --aug all --leads {L} --seed {s}", [PT_DONE]))
    # CPSC2018: 10-fold CV models (ensembled on the held-out test set by src.cpsc)
    for f in range(10):
        J.append(run(f"cpsc_pw_aug_f{f}", f"--init {PT} --aug all --fold {f}", [PT_DONE, CPSC], "cpsc2018"))
        J.append(run(f"cpsc_aug_f{f}", f"--aug all --fold {f}", [CPSC], "cpsc2018"))
    return J


def gpu_state():
    """Per GPU: free MiB and the set of compute PIDs."""
    q = lambda a: subprocess.run(["nvidia-smi", a, "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = [r.split(", ") for r in q("--query-gpu=index,uuid,memory.free").strip().splitlines()]
    uuid = {r[1]: int(r[0]) for r in rows}
    free = [int(r[2]) for r in rows]
    pids = [set() for _ in rows]
    for line in q("--query-compute-apps=pid,gpu_uuid").strip().splitlines():
        pid, u = line.split(", ")
        pids[uuid[u]].add(int(pid))
    return free, pids


def owner(pid):
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None


def project_job(pid):
    """Name and load of a job of this project running outside this scheduler, else None."""
    try:
        if os.readlink(f"/proc/{pid}/cwd") != ROOT:
            return None
        args = open(f"/proc/{pid}/cmdline", "rb").read().decode().split("\0")
    except OSError:
        return None
    if "src.pretrain" in args:
        return "pretrain", CAP  # a whole GPU: every pretrained-init job waits for it
    if "src.train" in args and "--tag" in args:
        ds = args[args.index("--dataset") + 1] if "--dataset" in args else "chapman"
        return f"{ds}/{args[args.index('--tag') + 1]}", None
    return None


def main():
    pending = [j for j in jobs() if not os.path.exists(j.done)]
    running = []  # (job, gpu, Popen)
    free, _ = gpu_state()
    n_gpu = len(free)
    print(f"{len(pending)} pending jobs, {n_gpu} GPUs, capacity {CAP}/GPU", flush=True)
    while pending or running:
        pending = [j for j in pending if not os.path.exists(j.done)]
        for r in running[:]:
            if r[2].poll() is not None:
                running.remove(r)
                ok = os.path.exists(r[0].done)
                print(time.strftime("%H:%M:%S"), "done" if ok else f"FAILED rc={r[2].returncode}", r[0].name, flush=True)
        free, pids = gpu_state()
        mine = {r[2].pid for r in running}
        load = [0.0] * n_gpu
        for j, g, _ in running:
            load[g] += j.weight
        weights = {j.name: j.weight for j in jobs()}
        elsewhere = set()
        for g in range(n_gpu):
            for pid in pids[g]:
                if pid in mine:
                    continue
                if owner(pid) != os.getuid():
                    load[g] += FOREIGN
                elif (pj := project_job(pid)):  # pre-training, or jobs of an earlier scheduler
                    elsewhere.add(pj[0])
                    load[g] += pj[1] or weights.get(pj[0], 1.0)
        for j in [j for j in pending if j.ready() and j.name not in elsewhere]:
            g = min(range(n_gpu), key=lambda i: load[i])
            if load[g] + j.weight > CAP or free[g] < j.mem:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            p = subprocess.Popen("exec " + j.cmd, shell=True, env=env, stdout=open(j.log, "w"), stderr=subprocess.STDOUT)
            running.append((j, g, p))
            pending.remove(j)
            load[g] += j.weight
            free[g] -= j.mem
            print(time.strftime("%H:%M:%S"), f"start gpu{g} load {load[g]:.1f}", j.name, flush=True)
        if not running and pending and not any(j.ready() for j in pending):
            print("waiting for dependencies:", sorted({d for j in pending for d in j.deps if not os.path.exists(d)}), flush=True)
            time.sleep(300)
        time.sleep(20)
    print("ALL_JOBS_DONE", flush=True)


if __name__ == "__main__":
    main()
