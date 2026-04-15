"""
TTS Hub WebUI — 基于 Gradio 的模型管理界面
"""

import os
import sys
import argparse
from pathlib import Path

import gradio as gr

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent))

from core.detector import detect_model_type, list_model_dirs
from core.registry import get_adapter, list_adapters, is_supported
from core.adapter_base import TTSRequest


DEFAULT_MODEL_DIR = os.environ.get("TTS_HUB_MODEL_DIR", str(Path(__file__).parent / "models"))


def scan_models(model_dir: str) -> list[dict]:
    """扫描模型目录"""
    return list_model_dirs(model_dir)


def get_model_choices(model_dir: str) -> list[str]:
    """获取模型下拉框选项"""
    models = scan_models(model_dir)
    choices = []
    for m in models:
        d = m["detection"]
        status = "✅" if d["confidence"] == "high" else "⚠️" if d["confidence"] == "medium" else "❓"
        label = f'{status} {m["name"]} [{d["model_type"]}]'
        choices.append(label)
    return choices or ["（未找到模型）"]


def load_model_handler(model_dir: str, selection: str, device: str) -> str:
    """加载模型"""
    if "未找到" in selection or not selection:
        return "❌ 请先选择一个模型"

    # 从选择中提取模型名
    model_name = selection.split("]")[0].split(" ")[-1] if "]" in selection else selection
    # 实际模型名（去掉状态图标）
    for part in selection.split(" "):
        if part and part not in ("✅", "⚠️", "❓"):
            model_name = part
            break

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


def synthesize_handler(
    text: str,
    speaker: str,
    language: str,
    speed: float,
    model_type: str,
) -> tuple:
    """合成语音"""
    if not text.strip():
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
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, response.audio, response.sample_rate)
        info = f"✅ 合成完成 | 时长: {response.duration:.2f}s | 采样率: {response.sample_rate}Hz"
        return tmp.name, info
    except NotImplementedError as e:
        return None, f"⚠️ {e}"
    except Exception as e:
        return None, f"❌ 合成失败: {e}"


def detect_single_handler(model_dir: str, model_name: str) -> str:
    """手动检测单个模型目录"""
    path = str(Path(model_dir) / model_name)
    result = detect_model_type(path)
    import json
    return json.dumps(result, indent=2, ensure_ascii=False)


def build_ui(model_dir: str = DEFAULT_MODEL_DIR) -> gr.Blocks:
    """构建 Gradio 界面"""

    css = """
    .model-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin: 8px 0; }
    """

    with gr.Blocks(title="TTS Hub", theme=gr.themes.Soft(), css=css) as demo:

        gr.Markdown("# 🎙️ TTS Hub — 统一 TTS 模型管理")
        gr.Markdown("下载模型即用，自动检测架构，一键切换")

        with gr.Row():
            # === 左侧：模型管理 ===
            with gr.Column(scale=1):
                gr.Markdown("### 📦 模型管理")

                model_dir_input = gr.Textbox(
                    label="模型目录",
                    value=model_dir,
                    placeholder="模型文件夹路径",
                )

                with gr.Row():
                    scan_btn = gr.Button("🔄 扫描模型", variant="secondary")
                    auto_refresh = gr.Checkbox(label="自动刷新", value=False)

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

                # 已注册适配器列表
                gr.Markdown("### 🔌 已注册适配器")
                adapters_info = gr.JSON(
                    value=list_adapters(),
                    label="适配器列表",
                )

            # === 右侧：语音合成 ===
            with gr.Column(scale=2):
                gr.Markdown("### 🎤 语音合成")

                tts_text = gr.Textbox(
                    label="合成文本",
                    placeholder="请输入要合成的文本...",
                    lines=4,
                )

                with gr.Row():
                    tts_speaker = gr.Textbox(
                        label="说话人 / 参考音频",
                        placeholder="说话人 ID 或参考音频路径",
                    )
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

        # === 底部：模型检测工具 ===
        with gr.Accordion("🔍 模型检测工具", open=False):
            with gr.Row():
                detect_input = gr.Textbox(label="模型目录名", placeholder="输入 models/ 下的子目录名")
                detect_btn = gr.Button("检测")
            detect_output = gr.JSON(label="检测结果")

        # === 事件绑定 ===
        def refresh_models(dir_path):
            choices = get_model_choices(dir_path)
            return gr.update(choices=choices)

        scan_btn.click(
            fn=refresh_models,
            inputs=[model_dir_input],
            outputs=[model_dropdown],
        )

        def on_load(selection, device, model_dir):
            status = load_model_handler(model_dir, selection, device)
            # 提取 model_type
            mtype = ""
            for m in scan_models(model_dir):
                if m["name"] in selection:
                    mtype = m["detection"]["model_type"]
                    break
            return status, mtype

        load_btn.click(
            fn=on_load,
            inputs=[model_dropdown, device_dropdown, model_dir_input],
            outputs=[load_status, tts_model_type],
        )

        synthesize_btn.click(
            fn=synthesize_handler,
            inputs=[tts_text, tts_speaker, tts_language, tts_speed, tts_model_type],
            outputs=[audio_output, synth_status],
        )

        detect_btn.click(
            fn=detect_single_handler,
            inputs=[model_dir_input, detect_input],
            outputs=[detect_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="TTS Hub WebUI")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型目录路径")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
    args = parser.parse_args()

    # 确保模型目录存在
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    print(f"🎙️ TTS Hub 启动中...")
    print(f"📁 模型目录: {args.model_dir}")
    print(f"🌐 访问: http://localhost:{args.port}")

    demo = build_ui(args.model_dir)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
