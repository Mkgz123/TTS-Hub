"""
TTS Hub WebUI — 基于 Gradio 的模型管理界面
包含：模型扫描、加载、合成、下载、批量合成
"""

import os
import sys
import io
import json
import tempfile
import argparse
from pathlib import Path

import gradio as gr

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent))

from core.detector import detect_model_type, list_model_dirs
from core.registry import get_adapter, list_adapters, is_supported
from core.adapter_base import TTSRequest
from core.download_manager import DownloadManager, KNOWN_MODELS
from env_manager import (
    install_miniconda, create_env, create_env_stream, _run_cmd_stream,
    install_model_deps, env_exists,
    get_env_python, list_envs, remove_env, is_conda_available,
    run_code_in_env, get_env_pip, MODEL_REQUIREMENTS, startup_check,
    SHARED_REQUIREMENTS, _has_nvidia_gpu,
    _PIP_TORCH_CUDA_INDEX, _PIP_TORCH_CPU_INDEX,
)

def _clean_stderr(stderr: str, max_chars: int = 1500) -> str:
    """保留 stderr 中真正的错误信息：Traceback、Error 行 + 末尾关键内容"""
    if not stderr:
        return ""

    # 始终保留的关键词（真正的错误）
    keep_keywords = ("Traceback", "Error:", "Exception", "error:", "File \"",
                     "NotImplementedError", "RuntimeError", "ValueError",
                     "ImportError", "ModuleNotFound", "FileNotFound",
                     "OSError", "TypeError", "KeyError", "CUDA")

    # 过滤掉的噪音
    skip_keywords = ("FutureWarning", "DeprecationWarning", "UserWarning",
                     "RuntimeWarning", "WARNING:", "torch_dtype",
                     "HF_TOKEN", "unauthenticated", "rate limit",
                     "Loading weights:", "Using `chunk_length_s`",
                     "You are using a", "please upgrade", "end of life",
                     "Couldn't find ffmpeg", "defaulting to ffmpeg",
                     "experimental with seq2seq", "caveats",
                     "A new version of the following files was downloaded",
                     "Make sure to double-check they do not contain",
                     "To avoid downloading new versions of the code file")

    lines = stderr.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # 进度条行（含 %| 和 it/s）
        if "%|" in stripped and "it/s" in stripped:
            continue
        if any(kw in stripped for kw in skip_keywords):
            continue
        kept.append(stripped)

    if not kept:
        return stderr[-max_chars:]

    result = "\n".join(kept)
    # 如果太长，优先保留末尾（错误信息通常在最后）
    if len(result) > max_chars:
        result = "...\n" + result[-max_chars:]
    return result

DEFAULT_MODEL_DIR = os.environ.get("TTS_HUB_MODEL_DIR", (Path(__file__).parent / "models").as_posix())
REFERENCE_AUDIO_DIR = str(Path(__file__).parent / "reference_audios")

# 支持的参考音频格式
REF_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# 全局下载管理器
_download_mgr = None


def get_download_manager(model_dir: str) -> DownloadManager:
    global _download_mgr
    if _download_mgr is None or str(_download_mgr.model_dir) != model_dir:
        _download_mgr = DownloadManager(model_dir)
    return _download_mgr


def get_reference_audio_choices() -> list[str]:
    """扫描参考音频文件夹，返回可选列表"""
    ref_dir = Path(REFERENCE_AUDIO_DIR)
    if not ref_dir.exists():
        return []
    choices = []
    for f in sorted(ref_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in REF_AUDIO_EXTENSIONS:
            choices.append(str(f))
    return choices


def auto_install_deps(model_type: str) -> str:
    """尝试自动创建 conda 环境并安装依赖"""
    if not is_conda_available():
        # conda 不可用，尝试自动安装
        result = install_miniconda()
        if not result["success"]:
            return f"❌ {result['message']}\n请手动安装: https://docs.conda.io/en/latest/miniconda.html"

    if model_type not in MODEL_REQUIREMENTS:
        return ""

    # 检查环境是否已存在
    if env_exists(model_type):
        # 环境已存在，检查是否需要更新依赖
        return ""

    # 创建环境并安装依赖
    try:
        return create_env(model_type)
    except Exception as e:
        return f"❌ 环境创建失败: {e}"


def get_model_detection_info(model_dir: str, selection: str) -> str:
    """获取模型的检测详情文本"""
    models = scan_models(model_dir)
    for m in models:
        if m["name"] in selection:
            d = m["detection"]
            confidence = d.get("confidence", "none")
            method = d.get("method", "unknown")
            mtype = d.get("model_type", "unknown")
            error = d.get("error")
            missing = d.get("missing_files", [])

            icon = {"high": "✅", "medium": "⚠️", "low": "⚠️", "none": "❌"}.get(confidence, "❓")

            lines = [f"{icon} 检测结果: {mtype}"]
            lines.append(f"   置信度: {confidence}")
            lines.append(f"   方法: {method}")

            if confidence == "medium":
                lines.append("   ⚠️ 仅通过目录名猜测，建议手动确认")
            if confidence == "none":
                lines.append("   ❌ 无法识别模型架构")
            if error:
                lines.append(f"   错误: {error}")
            if missing:
                lines.append(f"   缺少文件: {', '.join(missing)}")

            # 检查 conda 环境是否已创建
            if mtype != "unknown" and mtype in MODEL_REQUIREMENTS:
                if not env_exists(mtype):
                    lines.append(f"   📦 需要创建 conda 环境 (ttshub-{mtype})")
                else:
                    lines.append(f"   ✅ conda 环境已就绪 (ttshub-{mtype})")

            return "\n".join(lines)

    return "❓ 未找到模型信息"


def download_from_url_handler(url: str, model_dir: str) -> str:
    """从自定义 URL 下载文件到模型目录（支持 HuggingFace 仓库链接）"""
    if not url or not url.strip():
        return "❌ 请输入下载链接"
    url = url.strip()

    # 检测 HuggingFace 仓库链接
    hf_match = _parse_huggingface_url(url)
    if hf_match:
        return _download_huggingface_repo(hf_match, model_dir)

    try:
        import urllib.request
        from urllib.parse import urlparse, unquote

        parsed = urlparse(url)
        # 从 URL 提取文件名
        path_part = unquote(parsed.path.rstrip("/"))
        filename = path_part.rsplit("/", 1)[-1] if "/" in path_part else path_part
        if not filename:
            filename = "downloaded_model"

        dest_dir = Path(model_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        # 下载文件
        print(f"⬇️ 正在下载: {url}")
        urllib.request.urlretrieve(url, str(dest_path))

        # 自动解压 .zip / .tar.gz / .tar / .7z
        if dest_path.suffix in (".zip", ".gz", ".tar", ".7z", ".rar"):
            extract_dir = dest_dir / dest_path.stem
            extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                if dest_path.suffix == ".zip":
                    import zipfile
                    with zipfile.ZipFile(str(dest_path), "r") as zf:
                        zf.extractall(str(extract_dir))
                elif dest_path.name.endswith((".tar.gz", ".tgz")):
                    import tarfile
                    with tarfile.open(str(dest_path), "r:gz") as tf:
                        tf.extractall(str(extract_dir))
                elif dest_path.suffix == ".tar":
                    import tarfile
                    with tarfile.open(str(dest_path), "r:") as tf:
                        tf.extractall(str(extract_dir))
                elif dest_path.suffix == ".7z":
                    import subprocess
                    subprocess.run(["7z", "x", f"-o{extract_dir}", str(dest_path)], check=True)
                else:
                    return f"✅ 文件已下载\n📁 {dest_path}\n⚠️ 不支持自动解压，请手动处理"

                # 删除压缩包
                dest_path.unlink()
                return f"✅ 下载并解压完成\n📁 {extract_dir}"
            except Exception as extract_err:
                return f"✅ 文件已下载，但解压失败: {extract_err}\n📁 {dest_path}"

        return f"✅ 下载完成\n📁 {dest_path}"
    except Exception as e:
        return f"❌ 下载失败: {e}"


def _parse_huggingface_url(url: str):
    """解析 HuggingFace URL，返回 repo_id 和可选 revision"""
    import re
    # 匹配 https://huggingface.co/owner/repo 或包含 /tree/branch 等
    m = re.match(r"https?://huggingface\.co/([^/]+/[^/]+?)(?:\.git)?(?:/|$)", url)
    if m:
        repo_id = m.group(1)
        # 提取 revision（/tree/branch_name）
        rev_match = re.search(r"/tree/([^/]+)", url)
        revision = rev_match.group(1) if rev_match else None
        return {"repo_id": repo_id, "revision": revision}
    return None


def _download_huggingface_repo(hf_info: dict, model_dir: str) -> str:
    """使用 huggingface_hub 下载 HuggingFace 仓库"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return "❌ 需要安装 huggingface_hub: pip install huggingface_hub"

    repo_id = hf_info["repo_id"]
    revision = hf_info.get("revision")
    safe_name = repo_id.replace("/", "__")
    local_dir = Path(model_dir) / safe_name

    try:
        print(f"⬇️ 从 HuggingFace 下载: {repo_id}")
        kwargs = {
            "repo_id": repo_id,
            "local_dir": str(local_dir),
            "local_dir_use_symlinks": False,
        }
        if revision:
            kwargs["revision"] = revision
            print(f"   分支: {revision}")

        snapshot_download(**kwargs)

        return f"✅ 下载完成\n📁 {local_dir}"
    except Exception as e:
        return f"❌ 下载失败: {e}"


# ============================================================
# 模型管理
# ============================================================

def scan_models(model_dir: str) -> list[dict]:
    return list_model_dirs(model_dir)


def get_model_choices(model_dir: str) -> list[str]:
    models = scan_models(model_dir)
    choices = []
    for m in models:
        d = m["detection"]
        status = "✅" if d["confidence"] == "high" else "⚠️" if d["confidence"] == "medium" else "❓"
        label = f'{status} {m["name"]} [{d["model_type"]}]'
        choices.append(label)
    return choices or ["（未找到模型）"]


def load_model_handler(model_dir: str, selection: str, device: str) -> str:
    if "未找到" in selection or not selection:
        return "❌ 请先选择一个模型"

    models = scan_models(model_dir)
    target = None
    for m in models:
        if m["name"] in selection:
            target = m
            break

    if not target:
        return f"❌ 未找到模型: {selection}"

    model_type = target["detection"]["model_type"]
    if model_type == "unknown":
        return "❌ 无法识别模型架构，请手动指定"

    # === Step 1: 确保 conda 环境存在 ===
    env_status = ""
    if not is_conda_available():
        # 尝试自动安装 miniconda
        miniconda_result = install_miniconda()
        if not miniconda_result["success"]:
            env_status = f"⚠️ conda 不可用: {miniconda_result['message']}"
    if not env_status and not env_exists(model_type):
        env_status = auto_install_deps(model_type)

    # === Step 2: 尝试在当前进程加载 ===
    adapter = get_adapter(model_type)
    if adapter:
        try:
            adapter.load_model(target["path"], device=device)
            features = adapter.get_supported_features()
            feat_str = " ".join(f"{'✅' if v else '❌'} {k}" for k, v in features.items())
            result = f"✅ 已加载: {adapter.display_name}\n📁 {target['path']}\n🎤 特性: {feat_str}"
            if env_status:
                result = f"📦 环境:\n{env_status}\n\n{result}"
            return result
        except (ImportError, RuntimeError):
            pass  # 当前进程缺少依赖或无 GPU，走 conda subprocess 路径
        except Exception as e:
            result = f"❌ 加载失败: {e}"
            if env_status:
                result = f"📦 环境:\n{env_status}\n\n{result}"
            return result

    # === Step 3: 通过 conda 环境 subprocess 验证加载 ===
    if is_conda_available() and env_exists(model_type):
        # 用 conda 环境的 Python 验证模型是否可以加载
        check_code = f"""
import sys, torch
sys.path.insert(0, '{Path(__file__).parent.as_posix()}')
from core.registry import get_adapter
adapter = get_adapter('{model_type}')
if adapter:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adapter.load_model('{Path(target["path"]).as_posix()}', device=device)
    print('LOAD_OK')
else:
    print('LOAD_FAIL: adapter not found')
"""
        try:
            stdout, stderr, rc = run_code_in_env(model_type, check_code, timeout=900)
            if rc == 0 and "LOAD_OK" in stdout:
                features = []
                for line in stdout.strip().split("\n"):
                    if line.startswith("FEATURES:"):
                        features = line.split(":", 1)[1]
                result = f"✅ 已加载 (conda: ttshub-{model_type})\n📁 {target['path']}"
                if env_status:
                    result = f"📦 环境:\n{env_status}\n\n{result}"
                return result
            else:
                err_msg = _clean_stderr(stderr) if stderr else stdout[:500]
                result = f"❌ 加载失败 (conda环境):\n{err_msg}"
                if env_status:
                    result = f"📦 环境:\n{env_status}\n\n{result}"
                return result
        except Exception as e:
            result = f"❌ conda 环境执行失败: {e}"
            if env_status:
                result = f"📦 环境:\n{env_status}\n\n{result}"
            return result

    # 两条路都走不通
    if env_status:
        return f"📦 环境:\n{env_status}\n\n❌ 无法加载模型，请检查依赖是否安装完成"
    return f"❌ 未找到 {model_type} 的适配器，请检查环境配置"


# ============================================================
# 语音合成
# ============================================================

def synthesize_handler(text, speaker, language, speed, gap, model_type):
    if not text or not text.strip():
        return None, "❌ 请输入文本"

    adapter = get_adapter(model_type)
    if adapter and adapter.is_loaded:
        request = TTSRequest(
            text=text,
            speaker=speaker if speaker else None,
            language=language,
            speed=speed,
        )
        try:
            response = adapter.synthesize(request)
            audio = response.audio
            if gap != 0:
                from core.audio_utils import adjust_sentence_gap
                audio = adjust_sentence_gap(audio, response.sample_rate, gap_ms=gap)
            import soundfile as sf
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio, response.sample_rate)
            info = f"✅ 合成完成 | 时长: {len(audio)/response.sample_rate:.2f}s | 采样率: {response.sample_rate}Hz"
            if gap != 0:
                info += f" | 句间间隙: {gap}ms"
            return tmp.name, info
        except Exception as e:
            return None, f"❌ 合成失败: {e}"

    # 当前进程无适配器 → 尝试 conda 环境 subprocess
    if is_conda_available() and env_exists(model_type):
        return _synthesize_via_conda(text, speaker, language, speed, gap, model_type)

    return None, "❌ 请先加载模型"


def _synthesize_via_conda(text, speaker, language, speed, gap, model_type):
    """通过 conda 环境的 subprocess 进行合成"""
    import soundfile as sf
    import base64

    # 创建临时输出文件
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = Path(tmp.name).as_posix()
    tmp.close()

    code = f"""
import sys, json, base64, tempfile
sys.path.insert(0, '{Path(__file__).parent.as_posix()}')

from core.registry import get_adapter
from core.adapter_base import TTSRequest
import soundfile as sf

adapter = get_adapter('{model_type}')
# 需要找到模型路径 — 从已加载状态或扫描
import os
model_dir = os.environ.get('TTS_HUB_MODEL_DIR', '{DEFAULT_MODEL_DIR}')

# 尝试找到最近使用的模型路径
from core.detector import list_model_dirs
models = list_model_dirs(model_dir)
target_path = None
for m in models:
    if m['detection']['model_type'] == '{model_type}':
        target_path = m['path']
        break

if not target_path:
    print('ERROR: no model found')
    sys.exit(1)

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
adapter.load_model(target_path, device=device)
request = TTSRequest(
    text={repr(text)},
    speaker={repr(speaker) if speaker else 'None'},
    language='{language}',
    speed={speed},
)
response = adapter.synthesize(request)
audio = response.audio
gap = {gap}
if gap != 0:
    from core.audio_utils import adjust_sentence_gap
    audio = adjust_sentence_gap(audio, response.sample_rate, gap_ms=gap)
sf.write('{output_path}', audio, response.sample_rate)
print('SYNTH_OK')
"""
    try:
        stdout, stderr, rc = run_code_in_env(model_type, code, timeout=900)
        if rc == 0 and "SYNTH_OK" in stdout:
            if os.path.exists(output_path):
                data, sr = sf.read(output_path)
                duration = len(data) / sr
                info = f"✅ 合成完成 (conda环境) | 时长: {duration:.2f}s | 采样率: {sr}Hz"
                if gap > 0:
                    info += f" | 句间间隙: {gap}ms"
                return output_path, info
        return None, f"❌ 合成失败 (rc={rc}):\n{_clean_stderr(stderr)}"
    except Exception as e:
        return None, f"❌ 合成异常: {e}"


# ============================================================
# 批量合成
# ============================================================

def batch_synthesize(texts, speaker, language, speed, gap, model_type):
    """批量合成多段文本"""
    if not texts or not texts.strip():
        return None, "❌ 请输入文本（每行一段）"

    adapter = get_adapter(model_type)
    if not adapter or not adapter.is_loaded:
        return None, "❌ 请先加载模型"

    lines = [l.strip() for l in texts.strip().split("\n") if l.strip()]
    if not lines:
        return None, "❌ 没有有效文本"

    import soundfile as sf

    results = []
    total_duration = 0.0
    failed = 0

    for i, line in enumerate(lines):
        request = TTSRequest(
            text=line,
            speaker=speaker if speaker else None,
            language=language,
            speed=speed,
        )
        try:
            response = adapter.synthesize(request)
            results.append(response.audio)
            total_duration += response.duration
        except Exception as e:
            failed += 1
            print(f"批量合成第 {i+1} 段失败: {e}")

    if not results:
        return None, "❌ 所有段落合成失败"

    # 拼接所有音频
    import numpy as np
    combined = np.concatenate(results)
    sample_rate = adapter.default_sample_rate
    if gap != 0:
        from core.audio_utils import adjust_sentence_gap
        combined = adjust_sentence_gap(combined, sample_rate, gap_ms=gap)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, combined, sample_rate)

    info = (f"✅ 批量合成完成\n"
            f"📝 成功: {len(results)}/{len(lines)} 段\n"
            f"⏱️ 总时长: {total_duration:.2f}s")
    if gap != 0:
        info += f"\n🔧 句间间隙: {gap}ms"
    if failed:
        info += f"\n⚠️ 失败: {failed} 段"

    return tmp.name, info


# ============================================================
# 语音设计
# ============================================================

def voicegen_synthesize_handler(text, instruction, temperature, top_p, top_k,
                                 repetition_penalty, max_new_tokens, gap,
                                 model_type, model_dir, selection):
    if not text or not text.strip():
        return None, "❌ 请输入合成文本"
    if not instruction or not instruction.strip():
        return None, "❌ 请输入音色描述"
    if not selection or "未找到" in str(selection):
        return None, "❌ 请先选择并加载模型"

    # 解析模型路径
    model_path = ""
    for m in scan_models(model_dir):
        if m["name"] in str(selection):
            model_path = Path(m["path"]).as_posix()
            break
    if not model_path:
        return None, "❌ 未找到模型路径"

    import json as _json
    params = {
        "model_path": model_path,
        "text": text.strip(),
        "instruction": instruction.strip(),
        "audio_temperature": float(temperature),
        "audio_top_p": float(top_p),
        "audio_top_k": int(top_k),
        "audio_repetition_penalty": float(repetition_penalty),
        "max_new_tokens": int(max_new_tokens),
        "gap_ms": int(gap),
    }

    # 写参数到临时 JSON 文件
    params_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    _json.dump(params, params_file)
    params_file.close()

    # 输出 WAV 路径
    out_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = Path(out_file.name).as_posix()
    out_file.close()

    code = f'''
import json, sys
params_file = r"{Path(params_file.name).as_posix()}"
out_path = r"{out_path}"

with open(params_file, "r", encoding="utf-8") as f:
    params = json.load(f)

sys.path.insert(0, r"{Path(__file__).parent.as_posix()}")
from core.registry import get_adapter
adapter = get_adapter("{model_type}")
if not adapter:
    print("FAIL: adapter not found")
    sys.exit(1)

adapter.load_model(params["model_path"], device="cuda")
from core.adapter_base import TTSRequest
request = TTSRequest(
    text=params["text"],
    extra={{k: v for k, v in params.items() if k not in ("text", "model_path", "gap_ms")}},
)
response = adapter.synthesize(request)
audio = response.audio
gap_ms = params.get("gap_ms", 0)
if gap_ms != 0:
    from core.audio_utils import adjust_sentence_gap
    audio = adjust_sentence_gap(audio, response.sample_rate, gap_ms=gap_ms)
import soundfile as sf
sf.write(out_path, audio, response.sample_rate)
print(f"OK|{{response.duration:.2f}}|{{response.sample_rate}}")
'''

    try:
        stdout, stderr, rc = run_code_in_env(model_type, code, timeout=900)
        if rc == 0 and "OK|" in stdout:
            line = stdout.strip().split("\n")[-1]
            _, duration, sr = line.split("|")
            info = f"✅ 语音设计完成 | 时长: {float(duration):.2f}s | 采样率: {sr}Hz"
            if gap != 0:
                info += f" | 句间间隙: {gap}ms"
            # 清理 JSON 参数文件
            try:
                Path(params_file.name).unlink()
            except Exception:
                pass
            return out_path, info
        else:
            err = _clean_stderr(stderr) or stdout[:800]
            try:
                Path(params_file.name).unlink()
            except Exception:
                pass
            return None, f"❌ 语音设计失败:\n{err}"
    except Exception as e:
        try:
            Path(params_file.name).unlink()
        except Exception:
            pass
        return None, f"❌ 语音设计失败: {e}"


# ============================================================
# 多轮对话
# ============================================================

def chat_to_markdown(chat_state, spk_a_name, spk_b_name):
    """将对话状态渲染为聊天气泡 Markdown"""
    if not chat_state:
        return "*（对话记录将显示在这里）*"
    lines = []
    for i, msg in enumerate(chat_state):
        speaker = msg.get("speaker", "?")
        text = msg.get("text", "")
        name = spk_a_name if speaker == "A" else spk_b_name
        if speaker == "A":
            # 左对齐气泡（蓝色）
            lines.append(
                f'<div style="display:flex;align-items:flex-start;margin:8px 0;gap:8px">'
                f'<div style="background:#e3f2fd;border-radius:12px;padding:8px 14px;max-width:75%;">'
                f'<div style="font-size:0.8em;color:#1565c0;font-weight:bold;margin-bottom:2px;">{name}</div>'
                f'<div>{text}</div></div></div>'
            )
        else:
            # 右对齐气泡（绿色）
            lines.append(
                f'<div style="display:flex;align-items:flex-start;margin:8px 0;gap:8px;justify-content:flex-end;">'
                f'<div style="background:#e8f5e9;border-radius:12px;padding:8px 14px;max-width:75%;">'
                f'<div style="font-size:0.8em;color:#2e7d32;font-weight:bold;margin-bottom:2px;text-align:right;">{name}</div>'
                f'<div>{text}</div></div></div>'
            )
    return "".join(lines)


def multiturn_synthesize_handler(chat_state, spk_a_name, spk_b_name,
                                  spk_a_ref, spk_b_ref,
                                  temperature, top_p, top_k,
                                  repetition_penalty, repetition_window, gap,
                                  model_type, model_dir, selection):
    if not chat_state:
        return None, "❌ 请先添加对话消息"
    if not selection or "未找到" in str(selection):
        return None, "❌ 请先选择并加载模型"

    # 解析模型路径
    model_path = ""
    for m in scan_models(model_dir):
        if m["name"] in str(selection):
            model_path = Path(m["path"]).as_posix()
            break
    if not model_path:
        return None, "❌ 未找到模型路径"

    import json as _json

    # 构建文本列表和参考音频列表
    texts = []
    ref_audios = []
    for msg in chat_state:
        texts.append(msg.get("text", ""))
        speaker = msg.get("speaker", "A")
        ref = spk_a_ref if speaker == "A" else spk_b_ref
        ref_audios.append(ref if ref else None)

    params = {
        "model_path": model_path,
        "texts": texts,
        "ref_audios": ref_audios,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "repetition_penalty": float(repetition_penalty),
        "repetition_window": int(repetition_window),
        "gap_ms": int(gap),
    }

    # 写参数到临时 JSON 文件
    params_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    _json.dump(params, params_file)
    params_file.close()

    # 输出 WAV 路径
    out_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = Path(out_file.name).as_posix()
    out_file.close()

    code = f'''
import os as _os
_os.environ.setdefault("TORCHAUDIO_BACKEND", "soundfile")
import json, sys
params_file = r"{Path(params_file.name).as_posix()}"
out_path = r"{out_path}"

with open(params_file, "r", encoding="utf-8") as f:
    params = json.load(f)

sys.path.insert(0, r"{Path(__file__).parent.as_posix()}")
from core.registry import get_adapter
adapter = get_adapter("{model_type}")
if not adapter:
    print("FAIL: adapter not found")
    sys.exit(1)

adapter.load_model(params["model_path"], device="cuda")
from core.adapter_base import TTSRequest

# 将文本用换行连接，参考音频路径存在 extra 中
text = chr(10).join(params["texts"])
ref_audios = params.get("ref_audios", [])

request = TTSRequest(
    text=text,
    extra={{
        "ref_audio_list": ref_audios,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "repetition_penalty": params["repetition_penalty"],
        "repetition_window": params["repetition_window"],
    }},
)
response = adapter.synthesize(request)
audio = response.audio
gap_ms = params.get("gap_ms", 0)
if gap_ms != 0:
    from core.audio_utils import adjust_sentence_gap
    audio = adjust_sentence_gap(audio, response.sample_rate, gap_ms=gap_ms)
import soundfile as sf
sf.write(out_path, audio, response.sample_rate)
print(f"OK|{{response.duration:.2f}}|{{response.sample_rate}}")
'''

    try:
        stdout, stderr, rc = run_code_in_env(model_type, code, timeout=900)
        if rc == 0 and "OK|" in stdout:
            line = stdout.strip().split("\n")[-1]
            _, duration, sr = line.split("|")
            info = f"✅ 对话合成完成 | 时长: {float(duration):.2f}s | 采样率: {sr}Hz"
            if gap != 0:
                info += f" | 句间间隙: {gap}ms"
            try:
                Path(params_file.name).unlink()
            except Exception:
                pass
            return out_path, info
        else:
            err = _clean_stderr(stderr) or stdout[:800]
            try:
                Path(params_file.name).unlink()
            except Exception:
                pass
            return None, f"❌ 对话合成失败:\n{err}"
    except Exception as e:
        try:
            Path(params_file.name).unlink()
        except Exception:
            pass
        return None, f"❌ 对话合成失败: {e}"


# ============================================================
# 音效生成
# ============================================================

def soundeffect_synthesize_handler(prompt, temperature, top_p, top_k,
                                    repetition_penalty, max_tokens, gap,
                                    model_type, model_dir, selection):
    if not prompt or not prompt.strip():
        return None, "❌ 请输入音效描述"
    if not selection or "未找到" in str(selection):
        return None, "❌ 请先选择并加载模型"

    # 解析模型路径
    model_path = ""
    for m in scan_models(model_dir):
        if m["name"] in str(selection):
            model_path = Path(m["path"]).as_posix()
            break
    if not model_path:
        return None, "❌ 未找到模型路径"

    import json as _json
    params = {
        "model_path": model_path,
        "ambient_sound": prompt.strip(),
        "audio_temperature": float(temperature),
        "audio_top_p": float(top_p),
        "audio_top_k": int(top_k),
        "audio_repetition_penalty": float(repetition_penalty),
        "max_new_tokens": int(max_tokens),
        "gap_ms": int(gap),
    }

    # 写参数到临时 JSON 文件
    params_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    _json.dump(params, params_file)
    params_file.close()

    # 输出 WAV 路径
    out_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = Path(out_file.name).as_posix()
    out_file.close()

    code = f'''
import json, sys
params_file = r"{Path(params_file.name).as_posix()}"
out_path = r"{out_path}"

with open(params_file, "r", encoding="utf-8") as f:
    params = json.load(f)

sys.path.insert(0, r"{Path(__file__).parent.as_posix()}")
from core.registry import get_adapter
adapter = get_adapter("{model_type}")
if not adapter:
    print("FAIL: adapter not found")
    sys.exit(1)

adapter.load_model(params["model_path"], device="cuda")
from core.adapter_base import TTSRequest
request = TTSRequest(
    text=params["ambient_sound"],
    extra={{k: v for k, v in params.items() if k not in ("ambient_sound", "model_path", "gap_ms")}},
)
response = adapter.synthesize(request)
audio = response.audio
gap_ms = params.get("gap_ms", 0)
if gap_ms != 0:
    from core.audio_utils import adjust_sentence_gap
    audio = adjust_sentence_gap(audio, response.sample_rate, gap_ms=gap_ms)
import soundfile as sf
sf.write(out_path, audio, response.sample_rate)
print(f"OK|{{response.duration:.2f}}|{{response.sample_rate}}")
'''

    try:
        stdout, stderr, rc = run_code_in_env(model_type, code, timeout=900)
        if rc == 0 and "OK|" in stdout:
            line = stdout.strip().split("\n")[-1]
            _, duration, sr = line.split("|")
            info = f"✅ 音效生成完成 | 时长: {float(duration):.2f}s | 采样率: {sr}Hz"
            if gap != 0:
                info += f" | 句间间隙: {gap}ms"
            try:
                Path(params_file.name).unlink()
            except Exception:
                pass
            return out_path, info
        else:
            err = _clean_stderr(stderr) or stdout[:800]
            try:
                Path(params_file.name).unlink()
            except Exception:
                pass
            return None, f"❌ 音效生成失败:\n{err}"
    except Exception as e:
        try:
            Path(params_file.name).unlink()
        except Exception:
            pass
        return None, f"❌ 音效生成失败: {e}"


# ============================================================
# 模型下载
# ============================================================

def get_known_model_options():
    """获取已知模型的下拉选项"""
    options = []
    for mtype, models in KNOWN_MODELS.items():
        for repo_id, label in models.items():
            options.append(f"[{mtype}] {repo_id} — {label}")
    return options


def download_model_handler(selection: str, model_dir: str):
    """下载模型"""
    if not selection:
        return "❌ 请选择要下载的模型"

    # 解析 repo_id
    # 格式: "[model_type] repo_id — label"
    parts = selection.split(" ", 2)
    if len(parts) >= 2:
        model_type = parts[0].strip("[]")
        repo_id = parts[1]
    else:
        repo_id = selection
        model_type = None

    mgr = get_download_manager(model_dir)
    result = mgr.download_model(repo_id, model_type=model_type)

    if result["success"]:
        return f"✅ 下载完成\n📁 {result['path']}"
    else:
        return f"❌ {result['error']}"


def list_local_models_handler(model_dir: str):
    """列出本地模型"""
    mgr = get_download_manager(model_dir)
    models = mgr.list_local_models()

    if not models:
        return "📦 暂无本地模型"

    lines = []
    for m in models:
        size_str = f"{m['size_mb']:.0f} MB" if m['size_mb'] > 1 else f"{m['size_mb']*1024:.0f} KB"
        mtype = m.get("info", {}).get("model_type", "未知")
        lines.append(f"📁 {m['name']} | {size_str} | {mtype}")

    return "\n".join(lines)


def detect_single_handler(model_dir: str, model_name: str) -> str:
    path = str(Path(model_dir) / model_name)
    result = detect_model_type(path)
    return json.dumps(result, indent=2, ensure_ascii=False)


# ============================================================
# UI 构建
# ============================================================

def build_ui(model_dir: str = DEFAULT_MODEL_DIR) -> gr.Blocks:

    with gr.Blocks(title="TTS Hub") as demo:

        gr.Markdown("# 🎙️ TTS Hub — 统一 TTS 模型管理")
        gr.Markdown("下载模型即用，自动检测架构，一键切换")

        with gr.Tabs():
            # === Tab 1: 模型管理 + 合成 ===
            with gr.Tab("🏠 主面板"):
                with gr.Row():
                    # 左侧：模型管理
                    with gr.Column(scale=1):
                        gr.Markdown("### 📦 模型管理")

                        model_dir_input = gr.Textbox(
                            label="模型目录",
                            value=model_dir,
                            placeholder="模型文件夹路径",
                        )

                        with gr.Row():
                            scan_btn = gr.Button("🔄 扫描模型", variant="secondary")

                        model_dropdown = gr.Dropdown(
                            label="选择模型",
                            choices=get_model_choices(model_dir),
                            interactive=True,
                        )

                        detection_info = gr.Textbox(
                            label="模型检测详情",
                            value="请选择一个模型查看检测信息",
                            lines=5,
                            interactive=False,
                        )

                        device_dropdown = gr.Radio(
                            choices=["cuda", "cpu"],
                            value="cuda",
                            label="设备",
                        )

                        load_btn = gr.Button("🚀 加载模型", variant="primary")
                        load_status = gr.Textbox(label="状态", lines=4, interactive=False)

                        gr.Markdown("### 🔌 已注册适配器")
                        adapters_info = gr.JSON(
                            value=list_adapters(),
                            label="适配器列表",
                        )

                    # 右侧：语音合成
                    with gr.Column(scale=2):
                        gr.Markdown("### 🎤 语音合成")

                        tts_text = gr.Textbox(
                            label="合成文本",
                            placeholder="请输入要合成的文本...",
                            lines=4,
                        )

                        with gr.Row():
                            tts_speaker = gr.Dropdown(
                                label="参考音频",
                                choices=get_reference_audio_choices(),
                                value=None,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            refresh_ref_btn = gr.Button("🔄", scale=0, min_width=40)
                            tts_language = gr.Dropdown(
                                choices=["zh", "en", "ja", "ko", "fr", "de", "es"],
                                value="zh",
                                label="语言",
                            )
                            tts_speed = gr.Slider(
                                minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                                label="语速",
                            )
                            tts_gap = gr.Slider(
                                minimum=-2000, maximum=2000, value=0, step=50,
                                label="句间间隙 (ms, 0=不调整, 负数=缩减)",
                            )

                        tts_model_type = gr.Textbox(
                            label="当前模型类型",
                            value="",
                            interactive=False,
                        )

                        synthesize_btn = gr.Button("🔊 开始合成", variant="primary")
                        audio_output = gr.Audio(label="合成结果", type="filepath")
                        synth_status = gr.Textbox(label="合成信息", interactive=False)

            # === Tab 2: 批量合成 ===
            with gr.Tab("📚 批量合成"):
                gr.Markdown("### 批量文本合成")
                gr.Markdown("每行一段文本，按顺序合成并拼接")

                with gr.Row():
                    with gr.Column(scale=2):
                        batch_texts = gr.Textbox(
                            label="合成文本（每行一段）",
                            placeholder="第一段文本\n第二段文本\n第三段文本",
                            lines=10,
                        )
                    with gr.Column(scale=1):
                        batch_speaker = gr.Dropdown(
                            label="参考音频",
                            choices=get_reference_audio_choices(),
                            value=None,
                            interactive=True,
                            allow_custom_value=False,
                        )
                        batch_refresh_ref_btn = gr.Button("🔄 刷新参考音频列表")
                        batch_language = gr.Dropdown(
                            choices=["zh", "en", "ja", "ko"],
                            value="zh",
                            label="语言",
                        )
                        batch_speed = gr.Slider(
                            minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                            label="语速",
                        )
                        batch_gap = gr.Slider(
                            minimum=-2000, maximum=2000, value=0, step=50,
                            label="句间间隙 (ms, 0=不调整, 负数=缩减)",
                        )
                        batch_model_type = gr.Textbox(
                            label="当前模型类型",
                            value="",
                            interactive=False,
                        )

                batch_btn = gr.Button("🚀 批量合成", variant="primary")
                batch_audio = gr.Audio(label="批量合成结果", type="filepath")
                batch_status = gr.Textbox(label="合成信息", interactive=False)

            # === Tab 3: 模型下载 ===
            with gr.Tab("⬇️ 模型下载"):
                gr.Markdown("### 从 HuggingFace 下载模型")

                with gr.Row():
                    with gr.Column():
                        download_dir = gr.Textbox(
                            label="下载目录",
                            value=model_dir,
                        )
                        known_models_dropdown = gr.Dropdown(
                            label="选择模型",
                            choices=get_known_model_options(),
                            interactive=True,
                        )
                        download_btn = gr.Button("⬇️ 开始下载", variant="primary")
                        download_status = gr.Textbox(
                            label="下载状态",
                            lines=4,
                            interactive=False,
                        )

                        gr.Markdown("### 🔗 自定义下载链接")
                        gr.Markdown("输入模型直接下载地址（支持 .zip / .tar.gz 等压缩包自动解压）")
                        custom_url_input = gr.Textbox(
                            label="下载链接",
                            placeholder="https://example.com/model.zip",
                        )
                        custom_url_btn = gr.Button("🔗 从链接下载", variant="secondary")
                        custom_url_status = gr.Textbox(
                            label="下载状态",
                            lines=3,
                            interactive=False,
                        )

                    with gr.Column():
                        gr.Markdown("### 📂 本地模型")
                        local_refresh_btn = gr.Button("🔄 刷新列表")
                        local_models_display = gr.Textbox(
                            label="本地模型",
                            value=list_local_models_handler(model_dir),
                            lines=10,
                            interactive=False,
                        )

                        gr.Markdown("### 🎵 参考音频管理")
                        gr.Markdown(f"将参考音频文件放入 `{REFERENCE_AUDIO_DIR}` 目录")
                        ref_audio_display = gr.Textbox(
                            label="参考音频列表",
                            value="\n".join(get_reference_audio_choices()) or "（目录为空）",
                            lines=6,
                            interactive=False,
                        )
                        ref_refresh_btn = gr.Button("🔄 刷新参考音频列表")

            # === Tab 4: 语音设计 ===
            with gr.Tab("🎤 语音设计"):
                gr.Markdown("### 语音设计 — 用文字描述音色，无需参考音频")
                gr.Markdown("*仅需输入音色描述 + 合成文本，由 MOSS-VoiceGenerator 直接生成语音。需要 CUDA GPU。*")
                with gr.Row():
                    # 左侧：模型管理
                    with gr.Column(scale=1):
                        gr.Markdown("### 📦 模型管理")

                        vg_model_dropdown = gr.Dropdown(
                            label="选择模型",
                            choices=get_model_choices(model_dir),
                            interactive=True,
                        )

                        vg_detection_info = gr.Textbox(
                            label="模型检测详情",
                            value="请选择一个模型查看检测信息",
                            lines=5,
                            interactive=False,
                        )

                        vg_load_btn = gr.Button("🚀 加载模型", variant="primary")
                        vg_load_status = gr.Textbox(label="状态", lines=4, interactive=False)

                        vg_model_type = gr.Textbox(
                            label="当前模型类型",
                            value="",
                            interactive=False,
                        )

                    # 右侧：语音设计合成
                    with gr.Column(scale=2):
                        gr.Markdown("### 🎨 音色设计")

                        vg_instruction = gr.Textbox(
                            label="音色描述 (Instruction)",
                            placeholder="例如：一个温暖、轻柔的女声，语速缓慢、清晰。\n或者：一个疲惫、沙哑的老人声音，缓慢地抱怨。",
                            lines=4,
                        )

                        vg_text = gr.Textbox(
                            label="合成文本",
                            placeholder="请输入要用该音色朗读的文本内容...",
                            lines=5,
                        )

                        with gr.Accordion("采样参数", open=False):
                            with gr.Row():
                                vg_temperature = gr.Slider(
                                    minimum=0.1, maximum=3.0, value=1.5, step=0.05,
                                    label="audio_temperature",
                                )
                                vg_top_p = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.6, step=0.01,
                                    label="audio_top_p",
                                )
                            with gr.Row():
                                vg_top_k = gr.Slider(
                                    minimum=1, maximum=200, value=50, step=1,
                                    label="audio_top_k",
                                )
                                vg_repetition_penalty = gr.Slider(
                                    minimum=0.8, maximum=2.0, value=1.0, step=0.05,
                                    label="repetition_penalty",
                                )
                            vg_max_tokens = gr.Slider(
                                minimum=256, maximum=8192, value=4096, step=128,
                                label="max_new_tokens",
                            )
                            vg_gap = gr.Slider(
                                minimum=-2000, maximum=2000, value=0, step=50,
                                label="句间间隙 (ms, 0=不调整, 负数=缩减)",
                            )

                        vg_synthesize_btn = gr.Button("🎨 生成语音", variant="primary")
                        vg_audio_output = gr.Audio(label="合成结果", type="filepath")
                        vg_synth_status = gr.Textbox(label="合成信息", interactive=False)

            # === Tab 5: 音效生成 ===
            with gr.Tab("🔊 音效生成"):
                gr.Markdown("### 音效生成 — 用文字描述生成环境音效")
                gr.Markdown("*输入音效描述（如雨声、脚步声、鸟鸣等），由 MOSS-SoundEffect 直接生成。需要 CUDA GPU。*")
                with gr.Row():
                    # 左侧：模型管理
                    with gr.Column(scale=1):
                        gr.Markdown("### 📦 模型管理")

                        sf_model_dropdown = gr.Dropdown(
                            label="选择模型",
                            choices=get_model_choices(model_dir),
                            interactive=True,
                        )

                        sf_detection_info = gr.Textbox(
                            label="模型检测详情",
                            value="请选择一个模型查看检测信息",
                            lines=5,
                            interactive=False,
                        )

                        sf_load_btn = gr.Button("🚀 加载模型", variant="primary")
                        sf_load_status = gr.Textbox(label="状态", lines=4, interactive=False)

                        sf_model_type = gr.Textbox(
                            label="当前模型类型",
                            value="",
                            interactive=False,
                        )

                    # 右侧：音效合成
                    with gr.Column(scale=2):
                        gr.Markdown("### 🎧 音效描述")

                        sf_prompt = gr.Textbox(
                            label="音效描述",
                            placeholder="例如：雨声淅沥、远处有雷声隆隆。\n或者：清晨公园里的鸟鸣声，微风拂过树叶。",
                            lines=4,
                        )

                        with gr.Accordion("采样参数", open=False):
                            with gr.Row():
                                sf_temperature = gr.Slider(
                                    minimum=0.1, maximum=3.0, value=1.5, step=0.05,
                                    label="audio_temperature",
                                )
                                sf_top_p = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.6, step=0.01,
                                    label="audio_top_p",
                                )
                            with gr.Row():
                                sf_top_k = gr.Slider(
                                    minimum=1, maximum=200, value=50, step=1,
                                    label="audio_top_k",
                                )
                                sf_repetition_penalty = gr.Slider(
                                    minimum=0.8, maximum=2.0, value=1.2, step=0.05,
                                    label="repetition_penalty",
                                )
                            sf_max_tokens = gr.Slider(
                                minimum=256, maximum=8192, value=4096, step=128,
                                label="max_new_tokens",
                            )
                            sf_gap = gr.Slider(
                                minimum=-2000, maximum=2000, value=0, step=50,
                                label="句间间隙 (ms, 0=不调整, 负数=缩减)",
                            )

                        sf_synthesize_btn = gr.Button("🔊 生成音效", variant="primary")
                        sf_audio_output = gr.Audio(label="生成结果", type="filepath")
                        sf_synth_status = gr.Textbox(label="生成信息", interactive=False)

            # === Tab 6: 多轮对话 ===
            with gr.Tab("💬 多轮对话"):
                gr.Markdown("### 多轮对话 TTS 合成")
                gr.Markdown("*设置两个说话人，逐条添加对话消息，可选参考音频进行声音克隆。需要 CUDA GPU。*")
                with gr.Row():
                    # 左侧：模型管理 + 说话人设置
                    with gr.Column(scale=1):
                        gr.Markdown("### 📦 模型管理")
                        rt_model_dropdown = gr.Dropdown(
                            label="选择模型",
                            choices=get_model_choices(model_dir),
                            interactive=True,
                        )
                        rt_detection_info = gr.Textbox(
                            label="模型检测详情",
                            value="请选择一个模型查看检测信息",
                            lines=4,
                            interactive=False,
                        )
                        rt_load_btn = gr.Button("🚀 加载模型", variant="primary")
                        rt_load_status = gr.Textbox(label="状态", lines=3, interactive=False)
                        rt_model_type = gr.Textbox(
                            label="当前模型类型", value="", interactive=False,
                        )

                        gr.Markdown("---")
                        gr.Markdown("### 🎤 说话人设置")
                        with gr.Row():
                            rt_spk_a_name = gr.Textbox(label="说话人 A", value="A", placeholder="名字")
                            rt_spk_b_name = gr.Textbox(label="说话人 B", value="B", placeholder="名字")
                        with gr.Row():
                            rt_spk_a_ref = gr.Dropdown(
                                label="A 的参考音频（可选）",
                                choices=get_reference_audio_choices(),
                                interactive=True,
                            )
                            rt_spk_b_ref = gr.Dropdown(
                                label="B 的参考音频（可选）",
                                choices=get_reference_audio_choices(),
                                interactive=True,
                            )
                        rt_refresh_ref_btn = gr.Button("🔄 刷新参考音频", variant="secondary")

                    # 右侧：聊天合成
                    with gr.Column(scale=2):
                        gr.Markdown("### 💬 对话记录")
                        rt_chat_display = gr.Markdown(
                            value="*（对话记录将显示在这里）*",
                            elem_classes=["chat-container"],
                        )

                        with gr.Row():
                            rt_msg_input = gr.Textbox(
                                label="消息内容",
                                placeholder="输入对话内容...",
                                scale=3,
                            )
                            rt_msg_speaker = gr.Radio(
                                label="说话人",
                                choices=["A", "B"],
                                value="A",
                                scale=1,
                            )
                        with gr.Row():
                            rt_add_msg_btn = gr.Button("➕ 添加消息", variant="secondary")
                            rt_clear_chat_btn = gr.Button("🗑️ 清空对话", variant="secondary")

                        rt_chat_state = gr.State([])

                        with gr.Accordion("采样参数", open=False):
                            with gr.Row():
                                rt_temperature = gr.Slider(
                                    minimum=0.1, maximum=3.0, value=0.8, step=0.05,
                                    label="temperature",
                                )
                                rt_top_p = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.6, step=0.01,
                                    label="top_p",
                                )
                            with gr.Row():
                                rt_top_k = gr.Slider(
                                    minimum=1, maximum=200, value=30, step=1,
                                    label="top_k",
                                )
                                rt_repetition_penalty = gr.Slider(
                                    minimum=0.8, maximum=2.0, value=1.1, step=0.05,
                                    label="repetition_penalty",
                                )
                            rt_repetition_window = gr.Slider(
                                minimum=10, maximum=200, value=50, step=10,
                                label="repetition_window",
                            )
                            rt_gap = gr.Slider(
                                minimum=-2000, maximum=2000, value=0, step=50,
                                label="句间间隙 (ms, 0=不调整, 负数=缩减)",
                            )

                        rt_synthesize_btn = gr.Button("💬 合成对话", variant="primary")
                        rt_audio_output = gr.Audio(label="合成结果", type="filepath")
                        rt_synth_status = gr.Textbox(label="合成信息", interactive=False)

            # === Tab 7: 检测工具 ===
            with gr.Tab("🔍 检测工具"):
                with gr.Row():
                    detect_dir = gr.Textbox(
                        label="模型目录",
                        value=model_dir,
                    )
                    detect_name = gr.Textbox(
                        label="子目录名",
                        placeholder="输入 models/ 下的子目录名",
                    )
                    detect_btn = gr.Button("检测")
                detect_output = gr.JSON(label="检测结果")

            # === Tab 5: 环境管理 ===
            with gr.Tab("🔧 环境管理"):
                gr.Markdown("### Conda 环境管理")
                gr.Markdown("每个模型拥有独立 conda 环境，解决依赖冲突问题")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📋 环境状态")
                        conda_status_btn = gr.Button("🔄 刷新状态", variant="secondary")
                        conda_status_display = gr.Textbox(
                            label="Conda 状态",
                            value="点击刷新查看",
                            lines=4,
                            interactive=False,
                        )
                        env_list_display = gr.Textbox(
                            label="已创建环境",
                            value="点击刷新查看",
                            lines=10,
                            interactive=False,
                        )

                    with gr.Column():
                        gr.Markdown("### 🛠️ 环境操作")

                        # Conda 状态（仅未安装时显示安装按钮）
                        _conda_installed = is_conda_available()
                        if not _conda_installed:
                            gr.Markdown("⚠️ 未检测到 conda，请先安装")
                            install_conda_btn = gr.Button("📦 安装 Miniconda", variant="primary")
                        else:
                            install_conda_btn = gr.Button("📦 安装 Miniconda", visible=False)
                        install_conda_status = gr.Textbox(
                            label="",
                            lines=3,
                            interactive=False,
                            visible=not _conda_installed,
                        )

                        # 模型环境管理
                        gr.Markdown("### 模型环境")
                        env_model_type = gr.Dropdown(
                            label="选择模型类型",
                            choices=list(MODEL_REQUIREMENTS.keys()),
                            interactive=True,
                        )
                        with gr.Row():
                            env_create_btn = gr.Button("➕ 创建环境", variant="primary")
                            env_reinstall_btn = gr.Button("🔄 重装依赖")
                            env_remove_btn = gr.Button("🗑️ 删除环境", variant="stop")
                        env_action_log = gr.Textbox(
                            label="操作日志",
                            value="",
                            lines=18,
                            interactive=False,
                            max_lines=30,
                        )

                        gr.Markdown("### ⚡ 快速安装全部")
                        env_install_all_btn = gr.Button("🚀 安装全部环境", variant="primary")
                        env_install_all_status = gr.Textbox(
                            label="安装进度",
                            lines=6,
                            interactive=False,
                        )

        # ============================================================
        # 事件绑定
        # ============================================================

        # 主面板
        def refresh_models(dir_path):
            choices = get_model_choices(dir_path)
            return gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)

        scan_btn.click(
            fn=refresh_models,
            inputs=[model_dir_input],
            outputs=[model_dropdown, vg_model_dropdown, sf_model_dropdown, rt_model_dropdown],
        )

        # 模型选择变化时更新检测详情
        def on_model_select(selection, model_dir):
            info = get_model_detection_info(model_dir, selection)
            return info

        model_dropdown.change(
            fn=on_model_select,
            inputs=[model_dropdown, model_dir_input],
            outputs=[detection_info],
        )

        def on_load(selection, device, model_dir):
            status = load_model_handler(model_dir, selection, device)
            mtype = ""
            for m in scan_models(model_dir):
                if m["name"] in selection:
                    mtype = m["detection"]["model_type"]
                    break
            return status, mtype, mtype  # 同时更新 batch_model_type

        load_btn.click(
            fn=on_load,
            inputs=[model_dropdown, device_dropdown, model_dir_input],
            outputs=[load_status, tts_model_type, batch_model_type],
        )

        synthesize_btn.click(
            fn=synthesize_handler,
            inputs=[tts_text, tts_speaker, tts_language, tts_speed, tts_gap, tts_model_type],
            outputs=[audio_output, synth_status],
        )

        # 批量合成
        batch_btn.click(
            fn=batch_synthesize,
            inputs=[batch_texts, batch_speaker, batch_language, batch_speed, batch_gap, batch_model_type],
            outputs=[batch_audio, batch_status],
        )

        # 模型下载
        download_btn.click(
            fn=download_model_handler,
            inputs=[known_models_dropdown, download_dir],
            outputs=[download_status],
        )

        local_refresh_btn.click(
            fn=lambda d: list_local_models_handler(d),
            inputs=[download_dir],
            outputs=[local_models_display],
        )

        # 自定义链接下载
        custom_url_btn.click(
            fn=download_from_url_handler,
            inputs=[custom_url_input, download_dir],
            outputs=[custom_url_status],
        )

        # 参考音频刷新（主面板）
        refresh_ref_btn.click(
            fn=lambda: gr.update(choices=get_reference_audio_choices()),
            inputs=[],
            outputs=[tts_speaker],
        )

        # 参考音频刷新（批量面板）
        batch_refresh_ref_btn.click(
            fn=lambda: gr.update(choices=get_reference_audio_choices()),
            inputs=[],
            outputs=[batch_speaker],
        )

        # 参考音频管理面板刷新
        ref_refresh_btn.click(
            fn=lambda: "\n".join(get_reference_audio_choices()) or "（目录为空）",
            inputs=[],
            outputs=[ref_audio_display],
        )

        # 检测工具
        detect_btn.click(
            fn=detect_single_handler,
            inputs=[detect_dir, detect_name],
            outputs=[detect_output],
        )

        # ============================================================
        # 语音设计事件绑定
        # ============================================================

        # 语音设计模型下拉变化时更新检测详情
        def on_vg_model_select(selection, model_dir):
            return get_model_detection_info(model_dir, selection)

        vg_model_dropdown.change(
            fn=on_vg_model_select,
            inputs=[vg_model_dropdown, model_dir_input],
            outputs=[vg_detection_info],
        )

        # 语音设计模型加载
        def on_vg_load(selection, device, model_dir):
            status = load_model_handler(model_dir, selection, device)
            mtype = ""
            for m in scan_models(model_dir):
                if m["name"] in selection:
                    mtype = m["detection"]["model_type"]
                    break
            return status, mtype

        vg_load_btn.click(
            fn=on_vg_load,
            inputs=[vg_model_dropdown, device_dropdown, model_dir_input],
            outputs=[vg_load_status, vg_model_type],
        )

        # 语音设计合成
        vg_synthesize_btn.click(
            fn=voicegen_synthesize_handler,
            inputs=[vg_text, vg_instruction, vg_temperature, vg_top_p,
                    vg_top_k, vg_repetition_penalty, vg_max_tokens, vg_gap,
                    vg_model_type, model_dir_input, vg_model_dropdown],
            outputs=[vg_audio_output, vg_synth_status],
        )

        # ============================================================
        # 音效生成事件绑定
        # ============================================================

        # 音效模型下拉变化时更新检测详情
        def on_sf_model_select(selection, model_dir):
            return get_model_detection_info(model_dir, selection)

        sf_model_dropdown.change(
            fn=on_sf_model_select,
            inputs=[sf_model_dropdown, model_dir_input],
            outputs=[sf_detection_info],
        )

        # 音效模型加载
        def on_sf_load(selection, device, model_dir):
            status = load_model_handler(model_dir, selection, device)
            mtype = ""
            for m in scan_models(model_dir):
                if m["name"] in selection:
                    mtype = m["detection"]["model_type"]
                    break
            return status, mtype

        sf_load_btn.click(
            fn=on_sf_load,
            inputs=[sf_model_dropdown, device_dropdown, model_dir_input],
            outputs=[sf_load_status, sf_model_type],
        )

        # 音效生成
        sf_synthesize_btn.click(
            fn=soundeffect_synthesize_handler,
            inputs=[sf_prompt, sf_temperature, sf_top_p, sf_top_k,
                    sf_repetition_penalty, sf_max_tokens, sf_gap,
                    sf_model_type, model_dir_input, sf_model_dropdown],
            outputs=[sf_audio_output, sf_synth_status],
        )

        # ============================================================
        # 多轮对话事件绑定
        # ============================================================

        # 多轮对话模型下拉变化时更新检测详情
        def on_rt_model_select(selection, model_dir):
            return get_model_detection_info(model_dir, selection)

        rt_model_dropdown.change(
            fn=on_rt_model_select,
            inputs=[rt_model_dropdown, model_dir_input],
            outputs=[rt_detection_info],
        )

        # 多轮对话模型加载
        def on_rt_load(selection, device, model_dir):
            status = load_model_handler(model_dir, selection, device)
            mtype = ""
            for m in scan_models(model_dir):
                if m["name"] in selection:
                    mtype = m["detection"]["model_type"]
                    break
            return status, mtype

        rt_load_btn.click(
            fn=on_rt_load,
            inputs=[rt_model_dropdown, device_dropdown, model_dir_input],
            outputs=[rt_load_status, rt_model_type],
        )

        # 添加消息到对话
        def add_chat_message(chat_state, msg_text, speaker, spk_a_name, spk_b_name):
            if not msg_text or not msg_text.strip():
                return chat_state, chat_to_markdown(chat_state, spk_a_name, spk_b_name)
            chat_state = list(chat_state)  # 创建副本以触发 Gradio 更新
            chat_state.append({"speaker": speaker, "text": msg_text.strip()})
            return chat_state, chat_to_markdown(chat_state, spk_a_name, spk_b_name)

        rt_add_msg_btn.click(
            fn=add_chat_message,
            inputs=[rt_chat_state, rt_msg_input, rt_msg_speaker,
                    rt_spk_a_name, rt_spk_b_name],
            outputs=[rt_chat_state, rt_chat_display],
        )

        # 清空对话
        def clear_chat():
            return [], "*（对话记录将显示在这里）*"

        rt_clear_chat_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[rt_chat_state, rt_chat_display],
        )

        # 说话人名称变化时刷新显示
        def refresh_chat_display(chat_state, spk_a_name, spk_b_name):
            return chat_to_markdown(chat_state, spk_a_name, spk_b_name)

        rt_spk_a_name.change(
            fn=refresh_chat_display,
            inputs=[rt_chat_state, rt_spk_a_name, rt_spk_b_name],
            outputs=[rt_chat_display],
        )
        rt_spk_b_name.change(
            fn=refresh_chat_display,
            inputs=[rt_chat_state, rt_spk_a_name, rt_spk_b_name],
            outputs=[rt_chat_display],
        )

        # 多轮对话合成
        rt_synthesize_btn.click(
            fn=realtime_synthesize_handler,
            inputs=[rt_chat_state, rt_spk_a_name, rt_spk_b_name,
                    rt_spk_a_ref, rt_spk_b_ref,
                    rt_temperature, rt_top_p, rt_top_k,
                    rt_repetition_penalty, rt_repetition_window, rt_gap,
                    rt_model_type, model_dir_input, rt_model_dropdown],
            outputs=[rt_audio_output, rt_synth_status],
        )

        # 参考音频刷新（多轮对话面板）
        rt_refresh_ref_btn.click(
            fn=lambda: (gr.update(choices=get_reference_audio_choices()),
                        gr.update(choices=get_reference_audio_choices())),
            inputs=[],
            outputs=[rt_spk_a_ref, rt_spk_b_ref],
        )

        # ============================================================
        # 环境管理事件
        # ============================================================

        def refresh_env_status():
            """刷新 conda 状态和环境列表"""
            try:
                if is_conda_available():
                    from env_manager import find_conda
                    conda_info = f"✅ conda 可用\n路径: {find_conda()}"
                else:
                    conda_info = "❌ conda 未安装\n点击「创建环境」将自动安装 Miniconda"

                envs = list_envs()
                if envs:
                    lines = []
                    for e in envs:
                        lines.append(f"📁 {e['model_type']}  |  {e['packages_count']} 包  |  {e['path']}")
                    env_text = "\n".join(lines)
                else:
                    env_text = "暂无环境，选择模型类型后点击「创建环境」"

                return conda_info, env_text
            except Exception as e:
                return f"❌ 检查失败: {e}", "无法获取环境列表"

        conda_status_btn.click(
            fn=refresh_env_status,
            inputs=[],
            outputs=[conda_status_display, env_list_display],
        )

        def env_create_handler(model_type, progress=gr.Progress(track_tqdm=True)):
            if not model_type:
                yield "❌ 请选择模型类型"
                return
            try:
                # 先确保 conda 可用
                if not is_conda_available():
                    yield "📦 conda 未安装，正在自动安装 Miniconda..."
                    result = install_miniconda()
                    if not result["success"]:
                        yield f"❌ {result['message']}\n\n💡 建议: 点击上方「安装 Miniconda」按钮手动安装"
                        return
                    yield f"✅ Miniconda 安装完成\n\n---\n"

                # 检查是否已存在
                if env_exists(model_type):
                    yield f"⏭️ 环境 ttshub-{model_type} 已存在，无需重复创建"
                    return

                # 使用流式创建
                log_lines = []
                for update in create_env_stream(model_type):
                    log_lines.append(update["line"])
                    # 构建进度条描述
                    step = update["step"]
                    total = update["total"]
                    pct = step / total if total > 0 else 0
                    progress(pct, desc=f"{update['icon']} [{step}/{total}] {update['phase']}")

                    # 实时输出日志（保留最近 40 行）
                    display = log_lines[-40:]
                    yield "\n".join(display)

                    if update["done"]:
                        return

            except Exception as e:
                yield f"❌ 创建失败: {e}"

        env_create_btn.click(
            fn=env_create_handler,
            inputs=[env_model_type],
            outputs=[env_action_log],
        )

        # 独立的 Miniconda 安装按钮
        def install_conda_handler():
            if is_conda_available():
                from env_manager import find_conda
                return f"✅ conda 已安装\n路径: {find_conda()}"

            result = install_miniconda()
            return result["message"]

        install_conda_btn.click(
            fn=install_conda_handler,
            inputs=[],
            outputs=[install_conda_status],
        )

        def env_reinstall_handler(model_type, progress=gr.Progress(track_tqdm=True)):
            if not model_type:
                yield "❌ 请选择模型类型"
                return
            try:
                if not is_conda_available():
                    yield "❌ conda 未安装"
                    return
                if not env_exists(model_type):
                    # 环境不存在，走创建流程
                    yield f"📦 环境不存在，将创建新环境...\n\n---\n"
                    log_lines = []
                    for update in create_env_stream(model_type):
                        log_lines.append(update["line"])
                        step, total = update["step"], update["total"]
                        progress(step / total if total > 0 else 0,
                                 desc=f"{update['icon']} [{step}/{total}] {update['phase']}")
                        yield "\n".join(log_lines[-40:])
                        if update["done"]:
                            return
                    return

                # 环境存在，重新安装依赖
                pip_cmd = get_env_pip(model_type)
                if not pip_cmd:
                    yield "❌ 找不到 pip"
                    return

                req = MODEL_REQUIREMENTS.get(model_type, {})
                pip_cuda_deps = req.get("pip_cuda", [])
                pip_deps = req.get("pip", [])
                all_pip = SHARED_REQUIREMENTS + pip_deps
                log_lines = []

                # PyTorch
                if pip_cuda_deps:
                    has_gpu = _has_nvidia_gpu()
                    torch_index = _PIP_TORCH_CUDA_INDEX if has_gpu else _PIP_TORCH_CPU_INDEX
                    label = "CUDA" if has_gpu else "CPU"
                    log_lines.append(f"🔥 升级 PyTorch ({label})...")
                    yield "\n".join(log_lines)
                    cmd = pip_cmd + ["install", "--upgrade", "--index-url", torch_index] + pip_cuda_deps
                    for line in _run_cmd_stream(cmd):
                        log_lines.append(f"  {line}")
                        yield "\n".join(log_lines[-30:])
                    log_lines.append(f"✅ PyTorch 升级完成")
                    yield "\n".join(log_lines)

                progress(0.5, desc="📚 安装 pip 依赖...")

                # pip
                if all_pip:
                    log_lines.append(f"📚 安装 pip 依赖 ({len(all_pip)} 个包)...")
                    yield "\n".join(log_lines)
                    cmd = pip_cmd + ["install", "--upgrade"] + all_pip
                    for line in _run_cmd_stream(cmd):
                        log_lines.append(f"  {line}")
                        yield "\n".join(log_lines[-30:])
                    log_lines.append(f"✅ pip 依赖安装完成")
                    yield "\n".join(log_lines)

                progress(1.0, desc="✅ 完成")
                log_lines.append(f"\n🎉 依赖重装完成: ttshub-{model_type}")
                yield "\n".join(log_lines)

            except Exception as e:
                yield f"❌ 操作失败: {e}"

        env_reinstall_btn.click(
            fn=env_reinstall_handler,
            inputs=[env_model_type],
            outputs=[env_action_log],
        )

        def env_remove_handler(model_type):
            if not model_type:
                return "❌ 请选择模型类型"
            try:
                result = remove_env(model_type)
                return result["message"]
            except Exception as e:
                return f"❌ 删除失败: {e}"

        env_remove_btn.click(
            fn=env_remove_handler,
            inputs=[env_model_type],
            outputs=[env_action_log],
        )

        def env_install_all_handler(progress=gr.Progress(track_tqdm=True)):
            """一键创建所有环境（流式）"""
            if not is_conda_available():
                yield "📦 conda 未安装，正在自动安装 Miniconda..."
                result = install_miniconda()
                if not result["success"]:
                    yield f"❌ {result['message']}"
                    return
                yield "✅ Miniconda 安装完成\n\n---\n"

            total_models = len(MODEL_REQUIREMENTS)
            log_lines = []

            for i, mtype in enumerate(MODEL_REQUIREMENTS):
                progress_pct = i / total_models
                progress(progress_pct, desc=f"📦 [{i+1}/{total_models}] {mtype}")

                if env_exists(mtype):
                    log_lines.append(f"⏭️ {mtype}: 环境已存在，跳过")
                    yield "\n".join(log_lines[-25:])
                    continue

                log_lines.append(f"\n{'='*40}")
                log_lines.append(f"📦 [{i+1}/{total_models}] 创建环境: {mtype}")
                log_lines.append(f"{'='*40}")
                yield "\n".join(log_lines[-25:])

                for update in create_env_stream(mtype):
                    log_lines.append(update["line"])
                    # 嵌套进度：外层是模型计数，内层是当前模型步骤
                    inner_pct = update["step"] / update["total"] if update["total"] > 0 else 0
                    sub_pct = (i + inner_pct) / total_models
                    progress(sub_pct, desc=f"📦 [{i+1}/{total_models}] {mtype}: {update['phase']}")
                    yield "\n".join(log_lines[-25:])

                    if update["done"] and not update["success"]:
                        log_lines.append(f"❌ {mtype} 创建失败")
                        break

            progress(1.0, desc="🎉 全部完成")
            log_lines.append(f"\n🎉 全部环境处理完成")
            yield "\n".join(log_lines[-30:])

        env_install_all_btn.click(
            fn=env_install_all_handler,
            inputs=[],
            outputs=[env_action_log],
        )

    return demo


def main():
    # 修复 Windows 终端 emoji 编码问题
    if sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(description="TTS Hub WebUI")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型目录路径")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
    args = parser.parse_args()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(REFERENCE_AUDIO_DIR).mkdir(parents=True, exist_ok=True)

    print(f"🎙️ TTS Hub 启动中...")

    # === 启动检查 ===
    print("🔍 检查系统环境...")
    status = startup_check()

    if status["conda_available"]:
        print(f"✅ conda 可用: {status['conda_path']}")
        envs = status["envs"]
        if envs:
            print(f"📦 已创建 {len(envs)} 个模型环境:")
            for e in envs:
                print(f"   ttshub-{e['model_type']}: {e['packages_count']} 包")
        else:
            print("📦 暂无模型环境，首次加载模型时将自动创建")
    else:
        print("⚠️ conda 未安装")
        print("   首次加载模型时将自动安装 Miniconda")

    for w in status.get("warnings", []):
        print(f"⚠️ {w}")

    print(f"📁 模型目录: {args.model_dir}")
    print(f"📁 参考音频: {REFERENCE_AUDIO_DIR}")
    print(f"🌐 访问: http://localhost:{args.port}")

    demo = build_ui(args.model_dir)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
