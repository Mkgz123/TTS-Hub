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

DEFAULT_MODEL_DIR = os.environ.get("TTS_HUB_MODEL_DIR", str(Path(__file__).parent / "models"))
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


def download_from_url_handler(url: str, model_dir: str) -> str:
    """从自定义 URL 下载文件到模型目录"""
    if not url or not url.strip():
        return "❌ 请输入下载链接"
    url = url.strip()

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

        # 如果是目录压缩包，下载后自动解压
        print(f"⬇️ 正在下载: {url}")
        urllib.request.urlretrieve(url, str(dest_path))

        # 自动解压 .zip / .tar.gz / .tar / .7z
        if dest_path.suffix in (".zip", ".gz", ".tar", ".7z", ".rar"):
            import shutil, subprocess
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

    adapter = get_adapter(model_type)
    if not adapter:
        return f"❌ 未找到 {model_type} 的适配器"

    try:
        adapter.load_model(target["path"], device=device)
        features = adapter.get_supported_features()
        feat_str = " ".join(f"{'✅' if v else '❌'} {k}" for k, v in features.items())
        return f"✅ 已加载: {adapter.display_name}\n📁 {target['path']}\n🎤 特性: {feat_str}"
    except Exception as e:
        return f"❌ 加载失败: {e}"


# ============================================================
# 语音合成
# ============================================================

def synthesize_handler(text, speaker, language, speed, model_type):
    if not text or not text.strip():
        return None, "❌ 请输入文本"

    adapter = get_adapter(model_type)
    if not adapter or not adapter.is_loaded:
        return None, "❌ 请先加载模型"

    request = TTSRequest(
        text=text,
        speaker=speaker if speaker else None,
        language=language,
        speed=speed,
    )

    try:
        response = adapter.synthesize(request)
        import soundfile as sf
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, response.audio, response.sample_rate)
        info = f"✅ 合成完成 | 时长: {response.duration:.2f}s | 采样率: {response.sample_rate}Hz"
        return tmp.name, info
    except Exception as e:
        return None, f"❌ 合成失败: {e}"


# ============================================================
# 批量合成
# ============================================================

def batch_synthesize(texts, speaker, language, speed, model_type):
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
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, combined, adapter.default_sample_rate)

    info = (f"✅ 批量合成完成\n"
            f"📝 成功: {len(results)}/{len(lines)} 段\n"
            f"⏱️ 总时长: {total_duration:.2f}s")
    if failed:
        info += f"\n⚠️ 失败: {failed} 段"

    return tmp.name, info


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

            # === Tab 4: 检测工具 ===
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

        # ============================================================
        # 事件绑定
        # ============================================================

        # 主面板
        def refresh_models(dir_path):
            return gr.update(choices=get_model_choices(dir_path))

        scan_btn.click(
            fn=refresh_models,
            inputs=[model_dir_input],
            outputs=[model_dropdown],
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
            inputs=[tts_text, tts_speaker, tts_language, tts_speed, tts_model_type],
            outputs=[audio_output, synth_status],
        )

        # 批量合成
        batch_btn.click(
            fn=batch_synthesize,
            inputs=[batch_texts, batch_speaker, batch_language, batch_speed, batch_model_type],
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

    print(f"🎙️ TTS Hub 启动中...")
    print(f"📁 模型目录: {args.model_dir}")
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
