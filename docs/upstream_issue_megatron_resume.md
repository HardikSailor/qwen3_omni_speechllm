# Draft issue for modelscope/ms-swift (not posted; for review)

**Title:** Megatron resume with optimizer fails when the resuming run passes `--no_save_optim true` (`KeyError: 'optimizer'` / NaN grads)

**Versions:** ms-swift 4.6.0.dev0 (commit 8ec0455, also unchanged on main as of 2026-09-28), megatron-core 0.16.1, Transformer Engine 2.13, torch 2.10+cu128, mcore-bridge 1.6.4. Model: Qwen3-Omni-30B-A3B-Instruct, LoRA (`--target_modules linear_qkv linear_proj`), 1×H200.

**Steps to reproduce:**
1. `megatron sft ... --tuner_type lora --train_iters 6 --save_steps 3 --no_save_optim false --merge_lora false --output_dir OUT`
2. Resume: `megatron sft ... --mcore_adapter OUT/vX/checkpoint-3 --finetune false --no_load_optim false --no_save_optim true`

**Result:**
- default (`dp_reshardable`) optimizer checkpoint: `KeyError: 'optimizer'` in `megatron/core/optimizer/distrib_optimizer.py` `load_state_dict` (`state_dict["optimizer"]["param_groups"]`);
- with `--use_distributed_optimizer false`: `KeyError: 'state'`;
- with `--dist_ckpt_optim_fully_reshardable true`: the load passes, but the first backward after resume fails with `found NaN in local grad norm for bucket #0`.

The optimizer state dict handed to `optimizer.load_state_dict` contains only `{'param_state_sharding_type': str}`.

**Cause:** `load_mcore_checkpoint` builds the sharded load template via `_generate_state_dict(args, ...)` with the resuming run's `args`. `_generate_state_dict` adds `state_dict['optimizer']` only `if not args.no_save_optim`, so with `--no_save_optim true` the optimizer entries are not requested from the checkpoint (see the `# TODO: check no_save_optim` above the call).

**Workaround:** pass `--no_save_optim false` when resuming. With it, losses after resume match the uninterrupted run (max |Δloss| 0.009 over 10 iterations).

**Suggested fix:** when loading, decide whether to include the optimizer from `not finetune and not no_load_optim and not ckpt_args.no_save_optim` (as `load_mcore_checkpoint` already computes for `gen_sd_optim`), not from the current run's `no_save_optim`. For example, pass an explicit `include_optimizer` flag to `_generate_state_dict`.
