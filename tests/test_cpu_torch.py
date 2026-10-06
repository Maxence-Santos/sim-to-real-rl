from src.utils.torch_runtime import assert_cpu_pytorch


def test_installed_torch_is_the_cpu_build():
    version = assert_cpu_pytorch()
    assert version.endswith("+cpu") or "+cpu" in version
