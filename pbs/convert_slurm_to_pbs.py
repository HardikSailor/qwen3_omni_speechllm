#!/usr/bin/env python3
"""ONE-OFF (legacy): slurm/*.sbatch -> pbs/*.pbs for NSCC (PBS, enroot, H100). Run from the repo root: python pbs/convert_slurm_to_pbs.py

Header: #SBATCH job-name / gres gpu count / time -> #PBS (14 CPUs and 235 GB per GPU, queue normal, project 13003558).
Body:   the Orion path block (P=, RUN=, MODEL=, PYTHONPATH=) -> `source nscc/env.sh`; mixes/st_v0.yaml -> $MIX (NSCC data root).
Skipped: test_enroot_2gpu.sbatch (Orion enroot check; nscc/ covers it).
"""
import re, sys, pathlib
if "--force" not in sys.argv:
    sys.exit("pbs/*.pbs are now hand-maintained (they were generated once from slurm/*.sbatch). This would overwrite them; pass --force if you really want that.")
ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV = "/scratch/users/astar/ares/sailorhb/git_repos/qwen3_omni_speechllm/nscc/env.sh"
for src in sorted((ROOT / "slurm").glob("*.sbatch")):
    if src.name.startswith("test_enroot"):
        continue
    text = src.read_text()
    name = re.search(r"--job-name=(\S+)", text).group(1)
    gpus = int(re.search(r"--gres=gpu:\w+:(\d+)", text).group(1))
    wall = re.search(r"--time=(\S+)", text).group(1)
    body = [l for l in text.splitlines()[1:] if not l.startswith("#SBATCH")]
    out = []
    for l in body:
        if re.match(r"P=/scratch/prj0000000234", l):
            out.append(f"source {ENV}   # sets P, C, MODEL, MIX, RUN, PYTHONPATH, data + container paths")
            continue
        if re.match(r"(RUN|MODEL)=", l) or l.startswith("export PYTHONPATH="):
            continue
        l = l.replace("sbatch slurm/", "qsub pbs/").replace(".sbatch", ".pbs")
        if not l.startswith("#"):
            l = l.replace("mixes/st_v0.yaml", "$MIX")
        l = l.replace("slurm/", "pbs/")
        out.append(l)
    hdr = ["#!/bin/bash", f"#PBS -N {name}", f"#PBS -l select=1:ngpus={gpus}:ncpus={14*gpus}:mem={235*gpus}GB",
           f"#PBS -l walltime={wall}", "#PBS -q normal", "#PBS -P 13003558", "#PBS -j oe",
           f"#PBS -o {ROOT}/outputs/pbs_logs/{src.stem}.log"]
    dst = ROOT / "pbs" / (src.stem + ".pbs")
    dst.write_text("\n".join(hdr + out) + "\n")
    dst.chmod(0o755)
    print("wrote", dst.relative_to(ROOT))
