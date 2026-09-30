# Running on NSCC (PBS, enroot, H100 80 GB)

Tested 2026-09-29/30 on `a2ap-dgx004` and `a2ap-dgx038` (1x H100 80 GB, driver 595, enroot 3.5.0) and a 2-GPU `qsub` job.
Paths, what the container holds, and how to recreate `pydeps`: `../container/README.md`.

- `env.sh`: `source nscc/env.sh` sets `P` (repo), `C`/`CONTAINER_HOME`, `MODEL`, `MIX` (`mixes/st_v0.yaml`, already pointing at NSCC's data),
  `RUN` (= `container/run_container.sh`) and `PYTHONPATH` (repo + `container/pydeps`). Every `pbs/*.pbs` script sources it.
- `smoke_1gpu_h100.sh`: 10-step HF LoRA + resume check, run by hand inside an interactive job.
- `audit_config_1gpu.sh`: checks that both trainers use what a `--config` YAML says (non-default values, 4 steps each, ~20 min);
  summary in `outputs/config_audit/summary.txt`.
- `node_run.sh`, `nccl_check.py`: the per-node launcher and the NCCL bandwidth check used by the multi-node job (below).
- `../pbs/*.pbs`: the job scripts. `qsub pbs/<name>.pbs`, or run the body in an interactive job with `bash pbs/<name>.pbs`.
  Headers use `-q normal -P 13003558`, 14 CPUs and 235 GB per GPU; override the queue with `qsub -q dedicated ...`.
  Training settings come from `configs/lora_ddp.yaml` / `configs/lora_megatron.yaml` (`--config`); each script lists only its
  differences (GPUs, EP, batch, steps). Change a recipe in `configs/`, not in the scripts.
  `slurm/*.sbatch` are the legacy Orion versions the `.pbs` files were derived from; `pbs/` is now edited by hand.

## Interactive GPU sessions

Run everything on a GPU node, not the login node. Example: `qsub -I -q aidev -P 13003558 -l select=1:ngpus=1:ncpus=14:mem=235GB -l walltime=12:00:00`.
Compute nodes have no route to PyPI (the container has no proxy either); download wheels on the login node.
The first model load in a job takes ~8 min (63 GB from the shared filesystem), later loads on the same node ~2 min (page cache).

## Dedicated-queue tests (8 GPUs on 1 node, 32 GPUs on 4 nodes)

`qsub -q dedicated ...` overrides the header. (2026-09-29: the `dedicated` queue's ACL did not list this user;
if `qsub` refuses, ask NSCC to add the account.)

| Test | Command | Walltime | What it does |
|---|---|---|---|
| 1 node x 8 GPUs | `qsub -q dedicated pbs/train_8gpu_both.pbs` | 3.5 h | HF DDP and Megatron EP=8, 60 steps each, resume from step 30, disjoint-partition check; then an exploratory FSDP2 run (`--fsdp fsdp2`, 30 steps, phase `F8A`, last). Summary: `outputs/train_8gpu/summary.txt` |
| 4 nodes x 8 GPUs | `qsub -q dedicated pbs/train_32gpu_4node.pbs` | 4 h | NCCL bandwidth check, then the same two trainers on 32 GPUs (global batch 32, EP=8 inside each node). Summary: `outputs/train_32gpu/summary.txt` |

Multi-node design: the job script runs on the first node; `pbsdsh` starts `nscc/node_run.sh` on every node, which turns the node list
into `NNODES / NODE_RANK / MASTER_ADDR` (`train_omni.py` already relaunches itself under torchrun with those). Logs: `<phase>.log` is rank 0,
`<phase>.node<K>.log` the other nodes. Each node unpacks the enroot image on its own `/raid` (15 s).

Read first: the **NCCL line** at the top of `summary.txt`. `busbw` of a few GB/s means NCCL fell back to TCP sockets (InfiniBand not visible
in the container): stop and check `NCCL_DEBUG=INFO`, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME` before trusting training speed.

Checked without the dedicated queue (2026-09-30): dry run of the 4-node launcher with a fake node list (ranks, master,
FQDN de-duplication), and a real one-node run of `node_run.sh` -> container -> `torch.distributed.run`. **Not checked:** `pbsdsh` itself,
IB from inside the container, these two scripts with more than 1 GPU per job.
Dry run: `DRY_RUN=1 PBS_NODEFILE=<file with 4 hostnames> PBS_JOBID=dry bash pbs/train_32gpu_4node.pbs`.
