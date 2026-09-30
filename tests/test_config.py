"""CPU tests for train_omni.expand_config (`--config run.yaml`). Run in an interactive job:
    source nscc/env.sh && $RUN python tests/test_config.py
Also checks that every pbs/ script array is its recipe (configs/lora_*.yaml) plus exactly the listed differences.
(When the scripts were switched to --config on 2026-09-30, the expanded flags were compared with the old explicit
arrays: identical except ms-swift defaults now written out, and experts_impl grouped_mm added to smoke_1gpu.)"""
import os, shlex, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from train_omni import expand_config


def flags(argv):
    """argv -> {flag: [values]}; a repeated flag keeps the last occurrence (as ms-swift does)."""
    out, key = {}, None
    for t in argv:
        if t.startswith('--'):
            key = t[2:].split('=')[0]
            out[key] = [t.split('=', 1)[1]] if '=' in t else []
        elif key is not None:
            out[key].append(t)
    return out


def write(text):
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
    f.write(text); f.close()
    return f.name


def test_expand_and_override():
    os.environ['OMNI_TEST_MODEL'] = '/m/snap'
    p = write('model: ${OMNI_TEST_MODEL}\nlearning_rate: 1.0e-4\ngradient_checkpointing: true\n'
              'target_modules: [linear_qkv, linear_proj]\nvllm_limit_mm_per_prompt: {audio: 1}\nsave_steps: null\n'
              'lora_rank: 16\n')
    f = flags(expand_config(['--config', p, '--lora_rank', '8', '--output_dir', 'o']))
    assert f['model'] == ['/m/snap']
    assert float(f['learning_rate'][0]) == 1e-4
    assert f['gradient_checkpointing'] == ['true']
    assert f['target_modules'] == ['linear_qkv', 'linear_proj']
    assert f['vllm_limit_mm_per_prompt'] == ['{"audio": 1}']
    assert 'save_steps' not in f                                   # null drops the key
    assert f['lora_rank'] == ['8'] and f['output_dir'] == ['o']    # command line replaces the config value
    argv = expand_config([f'--config={p}', '--lora_rank=4'])
    assert argv.count('--lora_rank=4') == 1 and '--lora_rank' not in argv   # dropped from the config, not duplicated


def test_target_modules_file():
    p = write('target_modules_file: lora_targets_thinker_attn_audio_proj.txt\n')
    names = open(os.path.join(REPO, 'lora_targets_thinker_attn_audio_proj.txt')).read().split()
    assert flags(expand_config(['--config', p]))['target_modules'] == names and len(names) == 194


def test_unset_env_var_fails():
    p = write('model: ${OMNI_NOT_SET_ANYWHERE}\n')
    try:
        expand_config(['--config', p])
    except SystemExit:
        return
    raise AssertionError('unset variable accepted')


def test_no_config_is_identity():
    argv = ['--mix', 'x.yaml', '--lr', '1e-4']
    assert expand_config(list(argv)) == argv


def script_array(script, name):
    """An array (HF=(...), MEG=(...), COMMON=(...)) of a pbs/ script, expanded by bash with nscc/env.sh sourced."""
    src = open(os.path.join(REPO, 'pbs', script)).read()
    start = src.index(f'\n{name}=(') + 1
    body = src[start:src.index(')\n', start) + 1]
    cmd = (f'source {REPO}/nscc/env.sh; cd {REPO}; AUDIO=(--audio_lora_layers top4 --audio_lora_modules attn_mlp); '
           f'{body}; printf "%s\\n" "${{{name}[@]}}"')
    return subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, check=True).stdout.split('\n')[:-1]


def cfg_flags(cfg, extra=()):
    return flags(expand_config(['--config', os.path.join(REPO, cfg), *extra]))


def test_train_8gpu_uses_the_recipes_unchanged():
    for name, cfg in (('HF', 'configs/lora_ddp.yaml'), ('MEG', 'configs/lora_megatron.yaml')):
        assert flags(expand_config(script_array('train_8gpu_both.pbs', name))) == cfg_flags(cfg), name


# Every array of the pbs/ scripts that builds on a recipe, with the differences it is meant to have (and nothing else).
SCRIPT_OVERRIDES = [
    ('smoke_1gpu.pbs', 'COMMON', 'ddp', ['--save_total_limit', '3']),
    ('smoke_audio_lora_2gpu.pbs', 'HF', 'ddp', ['--val_choose', '8', '--audio_lora_layers', 'top4', '--audio_lora_modules',
     'attn_mlp', '--save_total_limit', '1', '--max_steps', '20', '--save_steps', '20', '--eval_steps', '20']),
    ('smoke_audio_lora_2gpu.pbs', 'MEG', 'megatron', ['--val_choose', '8', '--audio_lora_layers', 'top4',
     '--audio_lora_modules', 'attn_mlp', '--expert_model_parallel_size', '2', '--global_batch_size', '8',
     '--no_save_optim', 'true', '--train_iters', '20', '--save_steps', '20', '--eval_steps', '20', '--finetune', 'true']),
    ('smoke_megatron_1gpu.pbs', 'COMMON', 'megatron', ['--expert_model_parallel_size', '1', '--global_batch_size', '4',
     '--save_total_limit', '3']),
    ('smoke_megatron_resume.pbs', 'COMMON', 'megatron', ['--expert_model_parallel_size', '1', '--global_batch_size', '4',
     '--save_total_limit', '3']),
    ('verify_megatron_optim_resume.pbs', 'COMMON', 'megatron', ['--expert_model_parallel_size', '1', '--global_batch_size', '4']),
    ('train_4gpu_both.pbs', 'HF', 'ddp', ['--gradient_accumulation_steps', '8']),
    ('train_4gpu_both.pbs', 'MEG', 'megatron', ['--expert_model_parallel_size', '4']),
    ('train_32gpu_4node.pbs', 'HF', 'ddp', ['--gradient_accumulation_steps', '1']),
    ('train_32gpu_4node.pbs', 'MEG', 'megatron', []),
]


def test_pbs_scripts_differ_from_recipes_only_as_intended():
    for script, name, cfg, extra in SCRIPT_OVERRIDES:
        got = flags(expand_config(script_array(script, name)))
        assert got == cfg_flags(f'configs/lora_{cfg}.yaml', extra), (script, name)


if __name__ == '__main__':
    fails = 0
    for n, fn in list(globals().items()):
        if n.startswith('test_') and callable(fn):
            try:
                fn(); print('ok   ', n)
            except (Exception, SystemExit) as e:
                fails += 1; print('FAIL ', n, repr(e)[:600])
    sys.exit(1 if fails else 0)
