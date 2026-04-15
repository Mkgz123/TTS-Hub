"""
适配器注册表
管理所有可用的 TTS 适配器，提供按 model_type 查找的功能
"""

import importlib
from typing import Optional
from core.adapter_base import BaseTTSAdapter


# model_type → (module_path, class_name)
_ADAPTER_REGISTRY: dict[str, tuple[str, str]] = {}

# 缓存已实例化的适配器
_adapter_cache: dict[str, BaseTTSAdapter] = {}


def register_adapter(model_type: str, module_path: str, class_name: str):
    """注册一个适配器"""
    _ADAPTER_REGISTRY[model_type] = (module_path, class_name)


def get_adapter(model_type: str) -> Optional[BaseTTSAdapter]:
    """获取适配器实例（懒加载）"""
    if model_type not in _ADAPTER_REGISTRY:
        return None

    if model_type not in _adapter_cache:
        module_path, class_name = _ADAPTER_REGISTRY[model_type]
        try:
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
            _adapter_cache[model_type] = adapter_class()
        except (ImportError, AttributeError) as e:
            print(f"[Registry] 加载适配器失败 {model_type}: {e}")
            return None

    return _adapter_cache[model_type]


def list_adapters() -> list[dict]:
    """列出所有已注册的适配器"""
    result = []
    for model_type, (module_path, class_name) in _ADAPTER_REGISTRY.items():
        adapter = get_adapter(model_type)
        result.append({
            "model_type": model_type,
            "module": module_path,
            "class": class_name,
            "available": adapter is not None,
            "display_name": adapter.display_name if adapter else model_type,
            "supported_languages": adapter.supported_languages if adapter else [],
        })
    return result


def is_supported(model_type: str) -> bool:
    """检查是否支持该模型类型"""
    return model_type in _ADAPTER_REGISTRY


# ============================================================
# 注册所有内置适配器
# ============================================================

# 延迟注册 — 只注册元信息，实际 import 在 get_adapter() 时发生
register_adapter("fish-speech", "core.adapters.fish_speech", "FishSpeechAdapter")
register_adapter("f5-tts", "core.adapters.f5_tts", "F5TTSAdapter")
register_adapter("chattts", "core.adapters.chattts", "ChatTTSAdapter")
register_adapter("cosyvoice", "core.adapters.cosyvoice", "CosyVoiceAdapter")
register_adapter("kokoro", "core.adapters.kokoro", "KokoroAdapter")
register_adapter("xtts", "core.adapters.xtts", "XTTSAdapter")
register_adapter("moss-tts", "core.adapters.moss_tts", "MossttsAdapter")
