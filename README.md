# TTS Hub

统一 TTS 模型管理 WebUI — 下载模型即用，自动检测架构，一键切换。

## 支持模型

| 模型 | 架构 | 语言 | 显存 |
|------|------|------|------|
| Fish-Speech | VQGAN + LLM | 中英 | 4-8 GB |
| F5-TTS | Flow Matching DiT | 中英 | 4-6 GB |
| ChatTTS | GPT + VAE | 中文 | 2-4 GB |
| CosyVoice | VITS + Flow | 中英 | 4-6 GB |
| Kokoro | StyleTTS2 | 英文 | 1-2 GB |
| XTTSv2 | GPT + HifiGAN | 中英日韩法德西 | 4-6 GB |
| MOSS-TTS | VITS-like | 中文 | 2-4 GB |
| GPT-SoVITS | GPT + SoVITS | 中英日 | 4-8 GB |

## 快速开始

```bash
# 安装基础依赖
pip install -r requirements.txt

# 启动 WebUI（默认端口 7860）
python webui.py

# 指定模型目录和端口
python webui.py --model-dir /path/to/models --port 8080
```

## WebUI 功能

- **主面板** — 扫描、加载模型，单段文本合成
- **批量合成** — 多段文本依次合成并拼接
- **模型下载** — 内置 HuggingFace 知名模型列表，一键下载
- **检测工具** — 丢进一个目录，自动识别是什么模型

## 下载模型

WebUI 的「模型下载」Tab 提供常用模型列表，点击即下载。

也可手动从 HuggingFace 下载后放到 `models/` 目录，系统会自动识别架构。

常见模型地址：
- Fish-Speech: `fish-speech/fish-speech-1.5`
- F5-TTS: `SWivid/F5-TTS`
- ChatTTS: `2noise/ChatTTS`
- CosyVoice: `FunAudioLLM/CosyVoice-300M`
- Kokoro: `hexgrad/Kokoro-82M`
- XTTSv2: `coqui/XTTS-v2`

## 模型适配器

每个适配器独立封装，自动尝试多个 import 路径。如果模型的 Python 包未安装，会给出具体的 `pip install` 命令。

```
core/adapters/
├── fish_speech.py      # fish_speech 包 + transformers fallback
├── f5_tts.py           # f5_tts.api.F5TTS
├── chattts.py          # ChatTTS.Chat
├── cosyvoice.py        # 需克隆 CosyVoice 仓库
├── kokoro.py           # kokoro-onnx 优先，PyTorch fallback
├── xtts.py             # TTS.api (Coqui)
├── moss_tts.py         # moss_tts 包 + torch 直接加载 fallback
└── gpt_sovits.py       # 需克隆 GPT-SoVITS 仓库
```

## 环境管理

不同模型依赖冲突时，可用 `env_manager.py` 创建独立虚拟环境：

```bash
# 为指定模型创建 venv 并安装依赖
python env_manager.py create fish-speech
python env_manager.py create kokoro

# 查看所有环境
python env_manager.py list

# 删除环境
python env_manager.py remove kokoro
```

## 项目结构

```
tts-hub/
├── webui.py                    # Gradio WebUI（4 个 Tab）
├── core/
│   ├── adapter_base.py         # 统一接口：TTSRequest / TTSResponse
│   ├── registry.py             # 适配器注册与懒加载
│   ├── detector.py             # 模型架构自动检测（3 级策略）
│   ├── download_manager.py     # HuggingFace 模型下载
│   └── adapters/               # 8 个模型适配器
├── env_manager.py              # 虚拟环境管理
├── models/                     # 模型存储（自动生成）
├── requirements.txt
└── README.md
```

## 自定义适配器

1. 在 `core/adapters/` 下新建 `your_model.py`
2. 继承 `BaseTTSAdapter`，实现 `load_model()` 和 `synthesize()`
3. 在 `core/registry.py` 注册：`register_adapter("your-model", "core.adapters.your_model", "YourAdapter")`

## License

MIT
