"""
模型架构自动检测器
通过多种策略识别 HuggingFace 模型目录的架构类型
"""

import json
import os
from pathlib import Path
from typing import Optional


# 文件指纹 — 特征文件名 → 模型类型
FILE_FINGERPRINTS = {
    "fish-speech": [
        {"files": ["codec.pth", "model.pth"], "dirs": []},
        {"files": ["firefly-gan-vq-fsq-8x1024-21hz-generator.pth"], "dirs": []},
    ],
    "f5-tts": [
        {"files": ["model_*.pt", "vocab.txt"], "dirs": []},
        {"files": ["vae.pt"], "dirs": ["emilia"]},
    ],
    "chattts": [
        {"files": ["asset.zip", "DVAE_full.pt"], "dirs": ["asset"]},
        {"files": ["GPT_weights_v2.pt"], "dirs": []},
    ],
    "cosyvoice": [
        {"files": ["cosyvoice.yaml"], "dirs": []},
        {"files": ["llm.pt", "speech_tokenizer.pt"], "dirs": []},
    ],
    "kokoro": [
        {"files": ["kokoro_v0_19.onnx"], "dirs": []},
        {"files": ["kokoro-v1_0.pth"], "dirs": []},
    ],
    "xtts": [
        {"files": ["model.pth", "config.json", "dvae.pth"], "dirs": []},
        {"files": ["mel_stats.json", "dvae.pth"], "dirs": []},
    ],
    "moss-tts": [
        {"files": ["semantic_codec.pth", "acoustic_codec.pth"], "dirs": []},
        {"files": ["moss.yaml"], "dirs": []},
    ],
    "gpt-sovits": [
        {"files": ["GPT_SoVITS/pretrained_models"], "dirs": ["GPT_SoVITS"]},
    ],
}

# config.json 中的 model_type 直接映射
CONFIG_MODEL_TYPE_MAP = {
    "cosyvoice": "cosyvoice",
    "cosyvoice2": "cosyvoice",
    "xtts": "xtts",
    "kokoro": "kokoro",
    "f5_tts": "f5-tts",
    "fish_speech": "fish-speech",
    "fish-speech": "fish-speech",
}

# architectures 字段关键词映射
ARCH_KEYWORDS = {
    "qwen2": "fish-speech",
    "cosyvoice": "cosyvoice",
    "xtts": "xtts",
    "kokoro": "kokoro",
}


def _read_config(model_dir: Path) -> Optional[dict]:
    """安全读取 config.json"""
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    return None


def _check_fingerprint(model_dir: Path, fingerprint: dict) -> bool:
    """检查文件指纹是否匹配"""
    # 检查特征文件
    for pattern in fingerprint.get("files", []):
        if "*" in pattern:
            # 简单通配符匹配
            prefix, suffix = pattern.split("*", 1)
            matches = list(model_dir.glob(f"*{suffix}"))
            if not matches:
                return False
        else:
            if not (model_dir / pattern).exists():
                return False
    # 检查特征目录
    for dirname in fingerprint.get("dirs", []):
        if not (model_dir / dirname).is_dir():
            return False
    return True


def detect_model_type(model_dir: str) -> dict:
    """检测模型目录的架构类型

    Args:
        model_dir: 模型目录路径

    Returns:
        dict: {
            "model_type": str,          # 检测到的模型类型
            "confidence": str,          # "high" | "medium" | "low"
            "method": str,              # 检测方法
            "config": dict | None,      # config.json 内容
            "needs_confirmation": bool  # 是否需要用户确认
        }
    """
    model_path = Path(model_dir)
    if not model_path.is_dir():
        return {
            "model_type": "unknown",
            "confidence": "none",
            "method": "error",
            "config": None,
            "needs_confirmation": True,
            "error": f"目录不存在: {model_dir}",
        }

    files = [f.name for f in model_path.iterdir()]
    config = _read_config(model_path)

    # === 策略 1：config.json 的 model_type 字段 ===
    if config:
        model_type = config.get("model_type", "")
        if model_type in CONFIG_MODEL_TYPE_MAP:
            return {
                "model_type": CONFIG_MODEL_TYPE_MAP[model_type],
                "confidence": "high",
                "method": f"config.json model_type={model_type}",
                "config": config,
                "needs_confirmation": False,
            }

        # architectures 字段
        architectures = config.get("architectures", [])
        for arch in architectures:
            arch_lower = arch.lower()
            for keyword, mtype in ARCH_KEYWORDS.items():
                if keyword in arch_lower:
                    return {
                        "model_type": mtype,
                        "confidence": "high",
                        "method": f"config.json architectures={arch}",
                        "config": config,
                        "needs_confirmation": False,
                    }

    # === 策略 2：文件指纹检测 ===
    for mtype, fingerprints in FILE_FINGERPRINTS.items():
        for fp in fingerprints:
            if _check_fingerprint(model_path, fp):
                return {
                    "model_type": mtype,
                    "confidence": "high",
                    "method": f"file fingerprint: {fp['files']}",
                    "config": config,
                    "needs_confirmation": False,
                }

    # === 策略 3：目录名猜测 ===
    dir_name = model_path.name.lower()
    name_hints = {
        "cosyvoice": "cosyvoice",
        "fish-speech": "fish-speech",
        "fish_speech": "fish-speech",
        "f5-tts": "f5-tts",
        "f5_tts": "f5-tts",
        "chattts": "chattts",
        "chat-tts": "chattts",
        "kokoro": "kokoro",
        "xtts": "xtts",
        "moss": "moss-tts",
        "gpt-sovits": "gpt-sovits",
        "gpt_sovits": "gpt-sovits",
    }
    for hint, mtype in name_hints.items():
        if hint in dir_name:
            return {
                "model_type": mtype,
                "confidence": "medium",
                "method": f"directory name hint: {hint}",
                "config": config,
                "needs_confirmation": True,
            }

    # === 兜底：无法识别 ===
    return {
        "model_type": "unknown",
        "confidence": "none",
        "method": "no match",
        "config": config,
        "needs_confirmation": True,
        "files": files[:20],  # 返回前 20 个文件名帮助调试
    }


def list_model_dirs(base_dir: str) -> list[dict]:
    """扫描目录下所有可能的模型目录

    Returns:
        list[dict]: [{"path": ..., "name": ..., "detection": ...}, ...]
    """
    base = Path(base_dir)
    if not base.is_dir():
        return []

    results = []
    for item in sorted(base.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            detection = detect_model_type(str(item))
            results.append({
                "path": str(item),
                "name": item.name,
                "detection": detection,
            })
    return results
