#!/bin/bash
# Optional libraries of the AudioBench-SEA ASR normalisers (third_party/audiobench_sea), for eval jobs only.
# Installed --no-deps into $CONTAINER_HOME/pydeps_eval, which pbs/eval_omni.pbs *appends* to PYTHONPATH, so nothing
# here can shadow a package of the container or of pydeps. Only the missing dependencies are listed (everything else,
# e.g. numpy, torch, transformers, scikit-learn, pandas, regex, joblib, editdistance, is in the image).
# Runs on the login node (Python 3.10): wheels are fetched for the container's cp311 / manylinux.
#   bash container/install_pydeps_eval.sh
set -eu
C=${CONTAINER_HOME:-/scratch/users/astar/ares/sailorhb/container}
W=$C/wheels_pydeps_eval
PIP=${PIP:-pip}
PLAT=(--python-version 3.11 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64)
#        library (normaliser)                                    its missing dependencies
PKGS=(num2words docopt                                           # id, ms, ta, en number words
      cn2an proces                                               # zh numbers
      pythainlp attacut fire termcolor nptyping ssg python-crfsuite   # th segmentation (attacut is the default)
      indic-nlp-library morfessor indic-numtowords               # ta
      malaya malaya-boilerplate dateparser tzlocal ftfy unidecode     # ms
      nemo-text-processing pynini sacremoses inflect typeguard wget)  # en / sgen (NeMo TN); cdifflib skipped:
                                                                 # only its audio-based normaliser imports it
mkdir -p $W/sdist
for p in "${PKGS[@]}"; do
  $PIP download -q --no-deps -d $W "${PLAT[@]}" --only-binary=:all: $p 2>/dev/null \
    || { $PIP download -q --no-deps --no-binary=:all: -d $W/sdist $p && $PIP wheel -q --no-deps -w $W $W/sdist/${p}-*; }
done
$PIP install -q --no-deps --upgrade --target $C/pydeps_eval "${PLAT[@]}" --only-binary=:all: $W/*.whl
ls $C/pydeps_eval
