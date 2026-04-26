# TTS Hub

统一 TTS 模型管理 WebUI — 下载模型即用，自动检测架构，一键切换。

## 支持模型

| 模型 | 类型 | 架构 | 语言 | 显存 |
|------|------|------|------|------|
| MOSS-TTSD v1.0 | TTS · 对话合成 | Transformer + Codec | 中英 | 4-6 GB |
| MOSS-TTS Nano | TTS · 轻量 CPU | Codec + LLM | 中英日韩等 20 种 | CPU 可用 |
| MOSS-VoiceGenerator | 语音设计 | Multimodal LLM (1.7B) | 中英 | ~8 GB |
| MOSS-SoundEffect | 音效 | Multimodal LLM (8B) | 中英 | ~16 GB |
| MOSS-TTS-Realtime | 多轮对话 | MossTTSRealtime (1.7B) | 中英日韩等 20 种 | ~8 GB |

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

- **主面板** — 扫描、加载 TTS 模型，单段文本合成
- **批量合成** — 多段文本依次合成并拼接
- **语音设计** — 用文字描述音色，无需参考音频（MOSS-VoiceGenerator）
- **音效生成** — 用文字描述生成环境音效（MOSS-SoundEffect）
- **多轮对话** — 多轮流式对话合成，可选声音克隆（MOSS-TTS-Realtime）
- **模型下载** — 内置 HuggingFace 知名模型列表，一键下载
- **环境管理** — 每个模型独立 conda 环境
- **检测工具** — 丢进一个目录，自动识别是什么模型

## 下载模型

WebUI 的「模型下载」Tab 提供常用模型列表，点击即下载。

也可手动从 HuggingFace 下载后放到 `models/` 目录，系统会自动识别架构。

常见模型地址：
- MOSS-TTSD v1.0: `OpenMOSS-Team/MOSS-TTSD-v1.0`
- MOSS-TTS Nano: `OpenMOSS-Team/MOSS-TTS-Nano-100M`
- MOSS-VoiceGenerator: `OpenMOSS-Team/MOSS-VoiceGenerator`
- MOSS-TTS-Realtime: `OpenMOSS-Team/MOSS-TTS-Realtime`

详细使用说明见 [docs/models/](docs/models/)

## 模型适配器

每个适配器独立封装，自动尝试多个 import 路径。如果模型的 Python 包未安装，会给出具体的 `pip install` 命令。

```
core/adapters/
├── moss_tts.py         # HuggingFace transformers (MOSS-TTSD v1.0)
├── moss_tts_nano.py    # HuggingFace transformers (Nano 100M)
├── moss_voicegen.py    # HuggingFace transformers (VoiceGenerator 1.7B)
├── moss_soundeffect.py # HuggingFace transformers (SoundEffect 8B)
└── moss_tts_realtime.py # MossTTSRealtime + 独立 codec (Realtime 1.7B)
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
├── webui.py                    # Gradio WebUI（8 个 Tab）
├── core/
│   ├── adapter_base.py         # 统一接口：TTSRequest / TTSResponse
│   ├── registry.py             # 适配器注册与懒加载
│   ├── detector.py             # 模型架构自动检测（3 级策略）
│   ├── download_manager.py     # HuggingFace 模型下载
│   └── adapters/               # 5 个模型适配器
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
