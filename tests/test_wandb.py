"""CPU tests for train_omni.setup_wandb (--wandb true): trainer arguments, WANDB_* environment, resume rules.
    python tests/test_wandb.py        (no GPU, no network, no wandb package needed)"""
import json, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import train_omni
from train_omni import WANDB_STATE_FILE, parse_omni_args, setup_wandb

KEYS = ('WANDB', 'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_NAME', 'WANDB_RUN_GROUP', 'WANDB_TAGS', 'WANDB_MODE', 'WANDB_DIR',
        'WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_NOTES', 'WANDB_LOG_MODEL', 'RANK')


def run(mode, argv, env=None):
    for k in KEYS:
        os.environ.pop(k, None)
    os.environ.update(env or {})
    omni, swift_argv = parse_omni_args(['--mix', 'm.yaml'] + argv)
    out = setup_wandb(mode, omni, swift_argv)
    return out, {k: os.environ.get(k) for k in KEYS if os.environ.get(k)}


def value(argv, name):
    i = argv.index(name)
    end = i + 1
    while end < len(argv) and not argv[end].startswith('--'):
        end += 1
    return argv[i + 1:end]


def ckpt_with_run(parent, run_id='abc123', name='H8A', group='g1'):
    ck = os.path.join(parent, 'v0-x', 'checkpoint-30')
    os.makedirs(ck)
    with open(os.path.join(ck, WANDB_STATE_FILE), 'w') as f:
        json.dump({'id': run_id, 'name': name, 'group': group}, f)
    return ck


def test_off_by_default():
    out, env = run('sft', ['--output_dir', '/tmp/x'])
    assert out == ['--output_dir', '/tmp/x'] and not env


def test_env_switch_like_old_scripts():
    d = tempfile.mkdtemp()
    out, env = run('sft', ['--output_dir', d + '/H8A'], {'WANDB': '1', 'WANDB_ENTITY': 'team'})
    assert value(out, '--report_to') == ['tensorboard', 'wandb'] and value(out, '--run_name') == ['H8A']
    assert env['WANDB_PROJECT'] == 'meralion_v4' and env['WANDB_ENTITY'] == 'team'
    assert env['WANDB_DIR'] == d + '/H8A' and env['WANDB_LOG_MODEL'] == 'false' and 'WANDB_RUN_ID' not in env
    _, env = run('sft', ['--wandb', 'true', '--output_dir', d + '/H8A'])
    assert env['WANDB_ENTITY'] == 'i2r-llm'   # default team


def test_megatron_args_and_explicit_report_to():
    d = tempfile.mkdtemp()
    out, env = run('megatron', ['--wandb', 'true', '--wandb_project', 'p', '--wandb_group', 'job1', '--report_to', 'swanlab',
                                '--output_dir', d + '/M8A', '--lr', '1e-4'])
    assert value(out, '--report_to') == ['swanlab', 'wandb']
    assert value(out, '--wandb_project') == ['p'] and value(out, '--wandb_exp_name') == ['M8A']
    assert '--run_name' not in out and env['WANDB_RUN_GROUP'] == 'job1'
    out, _ = run('sft', ['--wandb', 'true', '--report_to', 'none', '--output_dir', d + '/H8A', '--lr', '1e-4'])
    assert value(out, '--report_to') == ['wandb']   # the recipes say `report_to: none`


def test_restart_continues_the_run():
    d = tempfile.mkdtemp()
    ck = ckpt_with_run(d + '/H8A')
    _, env = run('sft', ['--wandb', 'true', '--output_dir', d + '/H8A', '--resume_from_checkpoint', ck],
                 {'WANDB_RUN_ID': 'stale', 'WANDB_NAME': ''})
    assert env['WANDB_RUN_ID'] == 'abc123' and env['WANDB_RESUME'] == 'allow'


def test_resume_elsewhere_is_a_new_run_in_the_group():
    d = tempfile.mkdtemp()
    ck = ckpt_with_run(d + '/M8A')
    out, env = run('megatron', ['--wandb', 'true', '--output_dir', d + '/M8B', '--mcore_adapter', ck, '--finetune', 'false'])
    assert 'WANDB_RUN_ID' not in env and env['WANDB_RUN_GROUP'] == 'g1' and 'abc123' in env['WANDB_NOTES']
    assert value(out, '--wandb_exp_name') == ['M8B']
    _, env = run('megatron', ['--wandb', 'true', '--wandb_resume', 'always', '--output_dir', d + '/M8B',
                              '--mcore_adapter', ck, '--finetune', 'false'])
    assert env['WANDB_RUN_ID'] == 'abc123'
    _, env = run('megatron', ['--wandb', 'true', '--output_dir', d + '/M8B', '--mcore_adapter', ck, '--finetune', 'true'])
    assert 'WANDB_NOTES' not in env   # finetune true starts from the weights: not a resume


def test_state_file_written_by_the_logging_rank():
    class Run:
        id, name, project, entity, group, url = 'r1', 'H8A', 'p', 'e', 'g', 'https://wandb.ai/x'
    class Wandb:
        run = Run()
    d = tempfile.mkdtemp()
    train_omni.save_wandb_state(d + '/none')        # no wandb module loaded -> nothing written
    assert not os.path.exists(d + '/none')
    sys.modules['wandb'] = Wandb
    try:
        train_omni.save_wandb_state(d + '/ck')
    finally:
        del sys.modules['wandb']
    assert json.load(open(os.path.join(d, 'ck', WANDB_STATE_FILE)))['id'] == 'r1'


if __name__ == '__main__':
    fails = 0
    for n, fn in list(globals().items()):
        if n.startswith('test_') and callable(fn):
            try:
                fn(); print('ok   ', n)
            except (Exception, SystemExit) as e:
                fails += 1; print('FAIL ', n, repr(e)[:600])
    sys.exit(1 if fails else 0)
