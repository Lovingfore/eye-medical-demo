from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "setup_windows.ps1"


def test_windows_setup_script_declares_portable_local_deployment_flow() -> None:
    """一键脚本必须覆盖跨设备启动 Web 推理所需的关键步骤。"""
    assert SCRIPT_PATH.exists(), "setup_windows.ps1 尚未创建"
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    required_fragments = [
        "$PSScriptRoot",
        "requirements-render.txt",
        ".venv",
        "idrid_resnet_model.json",
        "idrid_resnet_best_fp16.pt",
        "web/manage.py",
        "migrate",
        "check",
        "runserver",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in content]
    assert not missing, f"脚本缺少部署步骤: {missing}"


def test_windows_setup_script_has_no_machine_specific_python_path() -> None:
    """脚本不能依赖原开发机上的固定 Python 盘符。"""
    assert SCRIPT_PATH.exists(), "setup_windows.ps1 尚未创建"
    content = SCRIPT_PATH.read_text(encoding="utf-8").lower()
    assert "d:\\python\\python.exe" not in content
    assert "c:\\python\\python.exe" not in content
