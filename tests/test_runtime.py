from manga_hd_transfer.config import RuntimeConfig
from manga_hd_transfer.runtime import configure_runtime, device_info, runtime_summary, select_device


def test_runtime_cpu_policy_is_side_effect_safe():
    policy=configure_runtime(RuntimeConfig(device='cpu', cpu_thread_ratio=.5))
    assert policy['cpu_threads'] >= 1
    assert select_device('cpu') == 'cpu'
    info=device_info('cpu')
    assert info.selected == 'cpu'
    assert runtime_summary('cpu')['device']['selected'] == 'cpu'
