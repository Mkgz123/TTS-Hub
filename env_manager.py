"""
环境管理器 — 为每个模型管理独立的虚拟环境
解决不同模型之间的依赖冲突问题
"""

import os
import json
import subprocess
import venv
from pathlib import Path
from typing import Optional


ENVS_DIR = Path(__file__).parent / "envs"

# 模型 → pip 依赖列表
MODEL_REQUIREMENTS = {
    "fish-speech": [
        "fish-speech",
    ],
    "f5-tts": [
        "f5-tts",
    ],
    "chattts": [
        "ChatTTS",
    ],
    "cosyvoice": [
        # CosyVoice 需要克隆仓库安装
        "torch>=2.0",
        "torchaudio",
        "transformers",
        "onnxruntime",
    ],
    "kokoro": [
        "kokoro-onnx",
    ],
    "xtts": [
        "TTS",
    ],
    "moss-tts": [
        "torch>=2.0",
        "torchaudio",
        "transformers",
        "monotonic-align",
    ],
    "gpt-sovits": [
        "torch>=2.0",
        "torchaudio",
        "transformers",
        "gradio",
        "cn2an",
        "pypinyin",
        "pyopenjtalk",
        "g2p_en",
        "langid",
    ],
}

# 共享依赖（每个环境都装）
SHARED_REQUIREMENTS = [
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
]


def get_env_path(model_type: str) -> Path:
    """获取模型虚拟环境路径"""
    return ENVS_DIR / model_type


def env_exists(model_type: str) -> bool:
    """检查环境是否存在"""
    python = get_env_path(model_type) / "bin" / "python"
    return python.exists()


def get_env_python(model_type: str) -> Optional[str]:
    """获取环境的 Python 解释器路径"""
    python = get_env_path(model_type) / "bin" / "python"
    if python.exists():
        return str(python)
    return None


def create_env(model_type: str) -> str:
    """创建虚拟环境"""
    env_path = get_env_path(model_type)
    env_path.mkdir(parents=True, exist_ok=True)

    print(f"[EnvManager] 创建环境: {model_type} ...")
    venv.create(str(env_path), with_pip=True)

    # 安装共享依赖
    pip = str(env_path / "bin" / "pip")
    subprocess.run(
        [pip, "install", "--upgrade", "pip"],
        capture_output=True,
    )
    subprocess.run(
        [pip, "install"] + SHARED_REQUIREMENTS,
        capture_output=True,
    )

    return str(env_path)


def install_model_deps(model_type: str, upgrade: bool = False) -> subprocess.CompletedProcess:
    """安装模型依赖到虚拟环境"""
    if not env_exists(model_type):
        create_env(model_type)

    pip = str(get_env_path(model_type) / "bin" / "pip")
    requirements = MODEL_REQUIREMENTS.get(model_type, [])

    if not requirements:
        return subprocess.CompletedProcess([], 0, "无额外依赖", "")

    cmd = [pip, "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(requirements)

    print(f"[EnvManager] 安装依赖: {model_type} → {requirements}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[EnvManager] 安装失败: {result.stderr}")
    else:
        print(f"[EnvManager] 安装完成: {model_type}")

    return result


def run_in_env(model_type: str, script: str, args: list = None) -> subprocess.CompletedProcess:
    """在模型环境中运行 Python 脚本"""
    python = get_env_python(model_type)
    if not python:
        raise RuntimeError(f"环境不存在: {model_type}，请先调用 create_env()")

    cmd = [python, script] + (args or [])
    return subprocess.run(cmd, capture_output=True, text=True)


def list_envs() -> list[dict]:
    """列出所有已创建的环境"""
    if not ENVS_DIR.exists():
        return []

    results = []
    for item in sorted(ENVS_DIR.iterdir()):
        if item.is_dir():
            python = item / "bin" / "python"
            pip = item / "bin" / "pip"
            # 获取已安装包列表
            if pip.exists():
                result = subprocess.run(
                    [str(pip), "list", "--format=json"],
                    capture_output=True, text=True,
                )
                try:
                    packages = json.loads(result.stdout)
                except json.JSONDecodeError:
                    packages = []
            else:
                packages = []

            results.append({
                "model_type": item.name,
                "path": str(item),
                "python": str(python) if python.exists() else None,
                "packages_count": len(packages),
                "packages": [p["name"] for p in packages[:10]],
            })

    return results


def remove_env(model_type: str) -> bool:
    """删除虚拟环境"""
    import shutil
    env_path = get_env_path(model_type)
    if env_path.exists():
        shutil.rmtree(str(env_path))
        print(f"[EnvManager] 已删除环境: {model_type}")
        return True
    return False


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  python env_manager.py create <model_type>    # 创建环境并安装依赖")
        print("  python env_manager.py list                   # 列出所有环境")
        print("  python env_manager.py remove <model_type>    # 删除环境")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "create":
        model_type = sys.argv[2]
        install_model_deps(model_type)
    elif cmd == "list":
        envs = list_envs()
        if not envs:
            print("暂无环境")
        for e in envs:
            print(f"  {e['model_type']:15}  {e['packages_count']} packages  {e['path']}")
    elif cmd == "remove":
        model_type = sys.argv[2]
        remove_env(model_type)
    else:
        print(f"未知命令: {cmd}")
