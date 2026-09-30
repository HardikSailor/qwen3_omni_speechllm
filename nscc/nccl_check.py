"""Multi-node NCCL sanity check: all_reduce bus bandwidth + which transport NCCL used. Run under torchrun via node_run.sh.
Expect >= ~150 GB/s busbw inside a node, and for 4 nodes >= ~40 GB/s if InfiniBand works; a few GB/s means NCCL fell back to TCP sockets."""
import os, time, torch, torch.distributed as dist
dist.init_process_group('nccl')
r, w, lr = dist.get_rank(), dist.get_world_size(), int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(lr)
x = torch.ones(256 * 1024 * 1024, device='cuda', dtype=torch.bfloat16)  # 512 MB
for _ in range(5): dist.all_reduce(x)
torch.cuda.synchronize(); t = time.time(); n = 20
for _ in range(n): dist.all_reduce(x)
torch.cuda.synchronize(); dt = (time.time() - t) / n
if r == 0:
    bus = 2 * (w - 1) / w * x.numel() * 2 / dt / 1e9
    print(f'[nccl_check] world={w} nodes={w // torch.cuda.device_count()} all_reduce 512MB: {dt*1e3:.1f} ms, busbw {bus:.1f} GB/s', flush=True)
dist.barrier(); dist.destroy_process_group()
