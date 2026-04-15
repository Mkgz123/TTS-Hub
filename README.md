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
| XTTSv2 | GPT + HifiGAN | 多语言 | 4-6 GB |
| MOSS-TTS | VITS-like | 中文 | 2-4 GB |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 WebUI
python webui.py

# 指定模型目录
python webui.py --model-dir /path/to/models
```

## 架构

```
tts-hub/
├── webui.py                # Gradio 界面入口
├── core/
│   ├── detector.py         # 模型架构自动检测
│   ├── registry.py         # 适配器注册表
│   ├── adapter_base.py     # 统一接口抽象基类
│   └── adapters/           # 各模型适配器
│       ├── cosyvoice.py
│       ├── fish_speech.py
│       ├── f5_tts.py
│       ├── chattts.py
│       ├── kokoro.py
│       ├── xtts.py
│       └── moss_tts.py
├── env_manager.py          # 虚拟环境/依赖管理
└── models/                 # 模型存储目录
```

## License

MIT
