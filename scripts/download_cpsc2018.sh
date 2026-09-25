#!/bin/bash
# CPSC2018 training set (6,877 12-lead ECGs, 500 Hz, 9 classes) from the PhysioNet/CinC 2020
# challenge mirror. Files are fetched in parallel and every file is retried until complete.
#   nohup scripts/download_cpsc2018.sh > logs/download_cpsc2018.log 2>&1 &
cd "$(dirname "$0")/../data" || exit 1
mkdir -p cpsc2018 && cd cpsc2018
B=https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018
for g in g1 g2 g3 g4 g5 g6 g7; do
  curl -s --retry 20 "$B/$g/" | grep -o 'href="A[0-9]*\.\(hea\|mat\)"' | cut -d'"' -f2 | sed "s|^|$g/|"
done > files.txt
echo "$(wc -l < files.txt) files listed"
fetch() {
  [ -s "$1" ] && return
  mkdir -p "$(dirname "$1")"
  until curl -s -f --retry 20 -o "$1.part" "$B/$1"; do sleep 5; done
  mv "$1.part" "$1"
}
export -f fetch; export B
xargs -P 128 -I{} bash -c 'fetch {}' < files.txt
echo "$(find . -name '*.mat' | wc -l) records"
echo CPSC_DOWNLOAD_DONE
