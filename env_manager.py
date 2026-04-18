"""
环境管理器 — 基于 Conda 为每个模型管理独立环境
解决不同模型之间的依赖冲突问题
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional


ENVS_DIR = Path(__file__).parent / "envs"

# 跨平台兼容
_IS_WIN = sys.platform == "win32"
_CREATE_FLAGS = 0x08000000 if _IS_WIN else 0  # CREATE_NO_WINDOW for Windows

# 模型 → conda/pip 依赖列表
MODEL_REQUIREMENTS = {
    "fish-speech": {
        "python": "3.10",
        "conda": [],
        "pip": ["fish-speech"],
    },
    "f5-tts": {
        "python": "3.10",
        "conda": [],
        "pip": ["f5-tts"],
    },
    "chattts": {
        "python": "3.10",
        "conda": [],
        "pip": ["ChatTTS", "torch", "torchaudio"],
    },
    "cosyvoice": {
        "python": "3.10",
        "conda": ["ffmpeg"],
        "pip": [
            "torch>=2.0", "torchaudio", "transformers",
            "onnxruntime-gpu", "librosa", "soundfile",
        ],
        "post_install": "git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice {env_dir}/CosyVoice && cd {env_dir}/CosyVoice && {pip} install -r requirements.txt",
    },
    "kokoro": {
        "python": "3.10",
        "conda": [],
        "pip": ["kokoro-onnx"],
    },
    "xtts": {
        "python": "3.10",
        "conda": [],
        "pip": ["TTS"],
    },
    "moss-tts": {
        "python": "3.10",
        "conda": ["ffmpeg"],
        "pip": [
            "torch>=2.0", "torchaudio", "transformers",
            "safetensors", "PyYAML", "monotonic-align",
        ],
    },
    "gpt-sovits": {
        "python": "3.10",
        "conda": ["ffmpeg"],
        "pip": [
            "torch>=2.0", "torchaudio", "transformers",
            "gradio", "cn2an", "pypinyin", "langid",
        ],
        "post_install": "git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS {env_dir}/GPT-SoVITS && cd {env_dir}/GPT-SoVITS && {pip} install -r requirements.txt",
    },
}

# 共享依赖（每个环境都装）
SHARED_REQUIREMENTS = [
    "numpy", "scipy", "librosa", "soundfile", "gradio",
]

# Conda 可执行文件缓存
_conda_bin: Optional[str] = None


def find_conda() -> Optional[str]:
    """查找 conda 可执行文件"""
    global _conda_bin
    if _conda_bin:
        return _conda_bin

    bin_names = ["conda", "mamba", "micromamba"]

    # 1. PATH 中查找（跨平台）
    for name in bin_names:
        try:
            if _IS_WIN:
                result = subprocess.run(
                    ["where", name], capture_output=True, text=True,
                    creationflags=_CREATE_FLAGS,
                )
            else:
                result = subprocess.run(
                    ["which", name], capture_output=True, text=True,
                    creationflags=_CREATE_FLAGS,
                )
            if result.returncode == 0:
                _conda_bin = result.stdout.strip().split("\n")[0].strip()
                return _conda_bin
        except (FileNotFoundError, OSError):
            continue

    # 2. 常见安装路径搜索
    if _IS_WIN:
        search_bases = [
            Path.home(),
            Path(os.environ.get("USERPROFILE", "")),
            Path(os.environ.get("LOCALAPPDATA", "")),
            Path("C:/"),
            Path("D:/"),
        ]
        search_subdirs = [
            "miniconda3", "anaconda3", "miniforge3", "mambaforge",
            "Miniconda3", "Anaconda3",
            "ProgramData/miniconda3", "ProgramData/anaconda3",
        ]
        exe_names = ["conda.exe", "mamba.exe", "micromamba.exe"]
        bin_subdirs = ["Scripts", "condabin"]
    else:
        search_bases = [Path.home(), Path("/opt"), Path("/root"), Path("/usr/local")]
        search_subdirs = [
            "miniconda3", "anaconda3", "miniforge3", "mambaforge",
            "miniconda", "anaconda",
        ]
        exe_names = bin_names
        bin_subdirs = ["bin", "condabin"]

    for base in search_bases:
        if not base or not base.exists():
            continue
        for subdir in search_subdirs:
            for bin_sub in bin_subdirs:
                for exe_name in exe_names:
                    candidate = base / subdir / bin_sub / exe_name
                    try:
                        if candidate.exists():
                            _conda_bin = str(candidate)
                            return _conda_bin
                    except (PermissionError, OSError):
                        continue

    return None


def is_conda_available() -> bool:
    """检查 conda 是否可用"""
    return find_conda() is not None


def get_env_path(model_type: str) -> Path:
    """获取模型 conda 环境路径"""
    return ENVS_DIR / model_type


def get_conda_env_name(model_type: str) -> str:
    """生成 conda 环境名称"""
    return f"ttshub-{model_type}"


def env_exists(model_type: str) -> bool:
    """检查 conda 环境是否存在"""
    conda = find_conda()
    if not conda:
        return False

    result = subprocess.run(
        [conda, "env", "list", "--json"],
        capture_output=True, text=True,
        creationflags=_CREATE_FLAGS,
    )
    if result.returncode != 0:
        return False

    try:
        data = json.loads(result.stdout)
        env_name = get_conda_env_name(model_type)
        for env_path in data.get("envs", []):
            if Path(env_path).name == env_name or env_name in env_path:
                return True
    except (json.JSONDecodeError, KeyError):
        pass

    return False


def get_env_python(model_type: str) -> Optional[str]:
    """获取 conda 环境的 Python 路径"""
    conda = find_conda()
    if not conda:
        return None

    result = subprocess.run(
        [conda, "env", "list", "--json"],
        capture_output=True, text=True,
        creationflags=_CREATE_FLAGS,
    )
    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        env_name = get_conda_env_name(model_type)
        for env_path in data.get("envs", []):
            if env_name in env_path:
                # 跨平台查找 Python
                if _IS_WIN:
                    candidates = [
                        Path(env_path) / "python.exe",
                        Path(env_path) / "Scripts" / "python.exe",
                    ]
                else:
                    candidates = [Path(env_path) / "bin" / "python"]
                for python in candidates:
                    if python.exists():
                        return str(python)
    except (json.JSONDecodeError, KeyError):
        pass

    return None


def create_env(model_type: str, progress_callback=None) -> str:
    """创建 conda 环境

    Args:
        model_type: 模型类型
        progress_callback: fn(message: str) 进度回调

    Returns:
        环境创建结果描述
    """
    conda = find_conda()
    if not conda:
        raise RuntimeError("conda 未安装，请先安装 Miniconda/Anaconda")

    # 确保 TOS 已接受
    _accept_conda_tos(conda)

    req = MODEL_REQUIREMENTS.get(model_type, {})
    python_ver = req.get("python", "3.10")
    env_name = get_conda_env_name(model_type)

    def log(msg):
        print(f"[EnvManager] {msg}")
        if progress_callback:
            progress_callback(msg)

    # 1. 创建环境
    log(f"创建 conda 环境: {env_name} (Python {python_ver})")
    cmd = [conda, "create", "-n", env_name, f"python={python_ver}", "-y", "-q"]
    result = subprocess.run(cmd, capture_output=True, text=True,
                    creationflags=_CREATE_FLAGS,
                )
    if result.returncode != 0:
        raise RuntimeError(f"创建环境失败: {result.stderr[:500]}")

    # 2. 安装 conda 依赖
    conda_deps = req.get("conda", [])
    if conda_deps:
        log(f"安装 conda 依赖: {conda_deps}")
        cmd = [conda, "install", "-n", env_name] + conda_deps + ["-y", "-q"]
        subprocess.run(cmd, capture_output=True, text=True,
                    creationflags=_CREATE_FLAGS,
                )

    # 3. 安装 pip 依赖
    pip_deps = req.get("pip", [])
    all_pip = SHARED_REQUIREMENTS + pip_deps
    if all_pip:
        log(f"安装 pip 依赖 ({len(all_pip)} 个包)...")
        pip = get_env_pip(model_type)
        if pip:
            cmd = [pip, "install", "--quiet"] + all_pip
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                    creationflags=_CREATE_FLAGS,
                )
            if result.returncode != 0:
                log(f"pip 安装部分失败: {result.stderr[:300]}")

    # 4. 后置安装步骤
    post_install = req.get("post_install")
    if post_install:
        log(f"执行后置安装...")
        pip = get_env_pip(model_type)
        post_cmd = post_install.format(
            env_dir=str(get_env_path(model_type)),
            pip=pip or "pip",
        )
        subprocess.run(
            post_cmd, shell=True, capture_output=True, text=True, timeout=300,
            creationflags=_CREATE_FLAGS,
        )

    log(f"环境创建完成: {env_name}")
    return f"✅ 环境创建完成: {env_name}"


def get_env_pip(model_type: str) -> Optional[str]:
    """获取环境的 pip 路径"""
    python = get_env_python(model_type)
    if python:
        pip = Path(python).parent / "pip"
        if pip.exists():
            return str(pip)
        # fallback: python -m pip
        return python
    return None


def install_model_deps(model_type: str, upgrade: bool = False) -> dict:
    """安装/更新模型依赖

    Returns:
        {"success": bool, "message": str}
    """
    conda = find_conda()
    if not conda:
        return {"success": False, "message": "conda 未安装"}

    _accept_conda_tos(conda)

    if not env_exists(model_type):
        try:
            msg = create_env(model_type)
            return {"success": True, "message": msg}
        except Exception as e:
            return {"success": False, "message": str(e)}

    req = MODEL_REQUIREMENTS.get(model_type, {})
    pip_deps = req.get("pip", [])
    all_pip = SHARED_REQUIREMENTS + pip_deps

    if not all_pip:
        return {"success": True, "message": "无额外依赖"}

    pip = get_env_pip(model_type)
    if not pip:
        return {"success": False, "message": "找不到 pip"}

    cmd = [pip, "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(all_pip)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                    creationflags=_CREATE_FLAGS,
                )
    if result.returncode == 0:
        return {"success": True, "message": f"✅ 依赖安装完成: {model_type}"}
    else:
        return {"success": False, "message": f"安装失败: {result.stderr[:500]}"}


def run_in_env(model_type: str, script: str, args: list = None) -> subprocess.CompletedProcess:
    """在 conda 环境中运行 Python 脚本"""
    python = get_env_python(model_type)
    if not python:
        raise RuntimeError(f"环境不存在: {model_type}")

    cmd = [python, script] + (args or [])
    return subprocess.run(cmd, capture_output=True, text=True,
                    creationflags=_CREATE_FLAGS,
                )


def run_code_in_env(model_type: str, code: str) -> tuple:
    """在 conda 环境中执行 Python 代码

    Returns:
        (stdout: str, stderr: str, returncode: int)
    """
    python = get_env_python(model_type)
    if not python:
        return "", f"环境不存在: {model_type}", 1

    result = subprocess.run(
        [python, "-c", code],
        capture_output=True, text=True, timeout=120,
        creationflags=_CREATE_FLAGS,
    )
    return result.stdout, result.stderr, result.returncode


def list_envs() -> list[dict]:
    """列出所有已创建的环境"""
    conda = find_conda()
    if not conda:
        return []

    result = subprocess.run(
        [conda, "env", "list", "--json"],
        capture_output=True, text=True,
        creationflags=_CREATE_FLAGS,
    )
    if result.returncode != 0:
        return []

    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        return []

    results = []
    for env_path in data.get("envs", []):
        env_path = Path(env_path)
        # 只列出 ttshub- 开头的环境
        if "ttshub-" not in str(env_path):
            continue

        # 提取 model_type
        model_type = env_path.name.replace("ttshub-", "")

        # 查找 Python（跨平台）
        if _IS_WIN:
            candidates = [
                env_path / "python.exe",
                env_path / "Scripts" / "python.exe",
            ]
        else:
            candidates = [env_path / "bin" / "python"]
        python = None
        for c in candidates:
            if c.exists():
                python = c
                break

        # 获取已安装包
        packages = []
        if python:
            try:
                result = subprocess.run(
                    [str(python), "-m", "pip", "list", "--format=json"],
                    capture_output=True, text=True, timeout=30,
                    creationflags=_CREATE_FLAGS,
                )
                if result.returncode == 0 and result.stdout.strip():
                    packages = json.loads(result.stdout)
            except Exception:
                pass

        results.append({
            "model_type": model_type,
            "path": str(env_path),
            "python": str(python) if python else None,
            "packages_count": len(packages),
            "packages": [p.get("name", "?") for p in packages[:10]],
        })

    return results


def startup_check() -> dict:
    """启动时检查系统状态

    Returns:
        {
            "conda_available": bool,
            "conda_path": str|None,
            "envs": list,
            "warnings": list,
        }
    """
    warnings = []

    conda_path = find_conda()
    conda_ok = conda_path is not None

    if not conda_ok:
        warnings.append("conda 未安装，首次加载模型时将自动安装 Miniconda")

    envs = []
    if conda_ok:
        envs = list_envs()

    return {
        "conda_available": conda_ok,
        "conda_path": conda_path,
        "envs": envs,
        "warnings": warnings,
    }


def remove_env(model_type: str) -> dict:
    """删除 conda 环境"""
    conda = find_conda()
    if not conda:
        return {"success": False, "message": "conda 未安装"}

    env_name = get_conda_env_name(model_type)
    result = subprocess.run(
        [conda, "env", "remove", "-n", env_name, "-y", "-q"],
        capture_output=True, text=True,
        creationflags=_CREATE_FLAGS,
    )
    if result.returncode == 0:
        return {"success": True, "message": f"已删除环境: {env_name}"}
    else:
        return {"success": False, "message": f"删除失败: {result.stderr[:300]}"}


def install_miniconda(install_dir: str = None, progress_callback=None) -> dict:
    """自动安装 Miniconda

    Args:
        install_dir: 安装目录
        progress_callback: fn(message: str) 进度回调

    Returns:
        {"success": bool, "message": str, "path": str}
    """
    global _conda_bin

    def log(msg):
        print(f"[EnvManager] {msg}")
        if progress_callback:
            progress_callback(msg)

    if is_conda_available():
        return {"success": True, "message": "conda 已安装", "path": find_conda()}

    if install_dir is None:
        install_dir = str(Path.home() / "miniconda3")

    install_path = Path(install_dir)
    if install_path.exists():
        if _IS_WIN:
            conda_bin = install_path / "Scripts" / "conda.exe"
            if not conda_bin.exists():
                conda_bin = install_path / "condabin" / "conda.bat"
        else:
            conda_bin = install_path / "bin" / "conda"
        if conda_bin.exists():
            _conda_bin = str(conda_bin)
            return {"success": True, "message": "conda 已安装（不在 PATH 中）", "path": str(conda_bin)}

    import tempfile
    import platform

    arch = platform.machine()

    if _IS_WIN:
        if arch in ("AMD64", "x86_64"):
            url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        elif arch == "ARM64":
            url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-arm64.exe"
        else:
            return {"success": False, "message": f"不支持的架构: {arch}"}
    else:
        if arch == "x86_64":
            url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        elif arch == "aarch64":
            url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        else:
            return {"success": False, "message": f"不支持的架构: {arch}"}

    try:
        if _IS_WIN:
            installer = Path(tempfile.gettempdir()) / "miniconda_installer.exe"
        else:
            installer = Path(tempfile.gettempdir()) / "miniconda_installer.sh"

        if installer.exists():
            installer.unlink(missing_ok=True)

        # 下载
        log(f"⬇️ 下载 Miniconda: {url}")
        result = subprocess.run(
            ["curl", "-fsSL", "--connect-timeout", "15",
             "--max-time", "300", "-o", str(installer), url],
            capture_output=True, text=True, timeout=310,
            creationflags=_CREATE_FLAGS,
        )
        if result.returncode != 0 or not installer.exists():
            err = result.stderr.strip()[:300] if result.stderr else f"退出码: {result.returncode}"
            return {
                "success": False,
                "message": f"❌ Miniconda 下载失败\n错误: {err}\n\n可能原因:\n1. 网络连接问题\n2. 需要配置代理\n\n手动下载: {url}",
            }

        log(f"✅ 下载完成 ({installer.stat().st_size / 1024 / 1024:.1f} MB)")

        # 安装
        if _IS_WIN:
            log("📦 正在安装 Miniconda (Windows)...")
            result = subprocess.run(
                [str(installer), "/InstallationType=JustMe", "/AddToPath=0",
                 "/RegisterPython=0", "/S", f"/D={install_dir}"],
                capture_output=True, text=True, timeout=600,
                creationflags=_CREATE_FLAGS,
            )
            conda_bin = install_path / "Scripts" / "conda.exe"
            if not conda_bin.exists():
                conda_bin = install_path / "condabin" / "conda.bat"
        else:
            log("📦 正在安装 Miniconda (Linux)...")
            result = subprocess.run(
                ["bash", str(installer), "-b", "-p", install_dir],
                capture_output=True, text=True, timeout=300,
                creationflags=_CREATE_FLAGS,
            )
            conda_bin = install_path / "bin" / "conda"

            log("🔧 初始化 conda...")
            subprocess.run(
                [str(conda_bin), "init", "bash"],
                capture_output=True, timeout=60,
                creationflags=_CREATE_FLAGS,
            )

        installer.unlink(missing_ok=True)

        if result.returncode != 0:
            return {"success": False, "message": f"❌ 安装失败\n错误: {result.stderr[:300]}"}

        if not conda_bin.exists():
            return {"success": False, "message": f"❌ 安装完成但找不到 conda\n路径: {conda_bin}"}

        _conda_bin = str(conda_bin)

        # 接受 TOS（新版 conda 要求）
        _accept_conda_tos(str(conda_bin))

        log(f"✅ Miniconda 安装完成: {install_dir}")
        return {
            "success": True,
            "message": f"✅ Miniconda 安装完成\n路径: {install_dir}",
            "path": str(conda_bin),
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "message": "❌ 安装超时，请检查网络连接后重试"}
    except Exception as e:
        return {"success": False, "message": f"❌ 安装异常: {e}"}


def _accept_conda_tos(conda_bin: str):
    """接受 conda TOS（新版 conda 要求）"""
    tos_channels = [
        "https://repo.anaconda.com/pkgs/main",
        "https://repo.anaconda.com/pkgs/r",
        "https://repo.anaconda.com/pkgs/msys2",
    ]
    for channel in tos_channels:
        try:
            subprocess.run(
                [conda_bin, "tos", "accept", "--override-channels",
                 "--channel", channel],
                capture_output=True, text=True, timeout=30,
                creationflags=_CREATE_FLAGS,
            )
        except Exception:
            pass  # 旧版 conda 没有 tos 命令，忽略


def ensure_conda_tos_accepted():
    """确保 conda TOS 已接受"""
    conda = find_conda()
    if conda:
        _accept_conda_tos(conda)


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
        print("  python env_manager.py install-conda          # 安装 Miniconda")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "create":
        model_type = sys.argv[2]
        create_env(model_type)
    elif cmd == "list":
        envs = list_envs()
        if not envs:
            print("暂无环境")
        for e in envs:
            print(f"  {e['model_type']:15}  {e['packages_count']} packages  {e['path']}")
    elif cmd == "remove":
        model_type = sys.argv[2]
        result = remove_env(model_type)
        print(result["message"])
    elif cmd == "install-conda":
        result = install_miniconda()
        print(result["message"])
    else:
        print(f"未知命令: {cmd}")
