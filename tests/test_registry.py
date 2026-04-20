"""
注册表与检测器单元测试
"""

import sys
import json
import pytest
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Registry 测试
# ============================================================

class TestRegistry:
    """适配器注册表核心功能"""

    def test_list_adapters_returns_all(self):
        from core.registry import list_adapters, _ADAPTER_REGISTRY
        adapters = list_adapters()
        assert len(adapters) == len(_ADAPTER_REGISTRY)

    def test_list_adapters_fields(self):
        from core.registry import list_adapters
        adapters = list_adapters()
        for a in adapters:
            assert "model_type" in a
            assert "module" in a
            assert "class" in a
            assert "available" in a
            assert "display_name" in a
            assert "supported_languages" in a

    def test_is_supported_known_types(self):
        from core.registry import is_supported
        known = ["fish-speech", "f5-tts", "chattts", "cosyvoice",
                 "kokoro", "xtts", "moss-tts", "gpt-sovits"]
        for mt in known:
            assert is_supported(mt), f"{mt} 应该被支持"

    def test_is_supported_unknown(self):
        from core.registry import is_supported
        assert not is_supported("nonexistent-model")

    def test_get_adapter_unknown_returns_none(self):
        from core.registry import get_adapter
        assert get_adapter("nonexistent-model") is None


# ============================================================
# Detector 测试
# ============================================================

class TestDetector:
    """模型架构自动检测 — detect_model_type() 返回 dict"""

    def _model_type(self, result):
        """从检测结果中提取 model_type"""
        if isinstance(result, dict):
            return result.get("model_type")
        return result  # 兼容旧版返回字符串的情况

    def test_detect_empty_dir(self, tmp_path):
        from core.detector import detect_model_type
        result = detect_model_type(str(tmp_path))
        mt = self._model_type(result)
        assert mt in (None, "unknown"), f"空目录应返回 None 或 unknown，实际: {mt}"

    def test_detect_nonexistent_dir(self):
        from core.detector import detect_model_type
        result = detect_model_type("/nonexistent/path")
        mt = self._model_type(result)
        assert mt in (None, "unknown"), f"不存在的目录应返回 None 或 unknown，实际: {mt}"

    def test_detect_cosyvoice_by_config(self, tmp_path):
        """config.json 中 model_type=cosyvoice 应被识别"""
        config = {"model_type": "cosyvoice", "hidden_size": 512}
        (tmp_path / "config.json").write_text(json.dumps(config))
        from core.detector import detect_model_type
        result = detect_model_type(str(tmp_path))
        assert self._model_type(result) == "cosyvoice"

    def test_detect_f5_tts_by_config(self, tmp_path):
        config = {"model_type": "f5_tts"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        from core.detector import detect_model_type
        result = detect_model_type(str(tmp_path))
        assert self._model_type(result) == "f5-tts"

    def test_detect_xtts_by_config(self, tmp_path):
        config = {"model_type": "xtts"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        from core.detector import detect_model_type
        result = detect_model_type(str(tmp_path))
        assert self._model_type(result) == "xtts"

    def test_detect_by_fingerprint_chattts(self, tmp_path):
        """ChatTTS 文件指纹识别"""
        (tmp_path / "asset.zip").touch()
        (tmp_path / "DVAE_full.pt").touch()
        (tmp_path / "asset").mkdir()
        from core.detector import detect_model_type
        result = detect_model_type(str(tmp_path))
        assert self._model_type(result) == "chattts"

    def test_detect_by_fingerprint_moss(self, tmp_path):
        (tmp_path / "semantic_codec.pth").touch()
        (tmp_path / "acoustic_codec.pth").touch()
        from core.detector import detect_model_type
        result = detect_model_type(str(tmp_path))
        assert self._model_type(result) == "moss-tts"

    def test_detect_by_fingerprint_kokoro_onnx(self, tmp_path):
        (tmp_path / "kokoro_v0_19.onnx").touch()
        from core.detector import detect_model_type
        result = detect_model_type(str(tmp_path))
        assert self._model_type(result) == "kokoro"

    def test_detect_by_fingerprint_cosyvoice(self, tmp_path):
        (tmp_path / "cosyvoice.yaml").touch()
        from core.detector import detect_model_type
        result = detect_model_type(str(tmp_path))
        assert self._model_type(result) == "cosyvoice"

    def test_detect_invalid_json_config(self, tmp_path):
        """config.json 格式错误时应不崩溃，返回 unknown"""
        (tmp_path / "config.json").write_text("NOT VALID JSON {{{")
        from core.detector import detect_model_type
        result = detect_model_type(str(tmp_path))
        mt = self._model_type(result)
        assert mt in (None, "unknown"), f"格式错误应返回 None 或 unknown，实际: {mt}"

    def test_detect_by_architecture_qwen2(self, tmp_path):
        """architectures 含 Qwen2 时应识别为 fish-speech"""
        config = {"architectures": ["Qwen2ForCausalLM"]}
        (tmp_path / "config.json").write_text(json.dumps(config))
        from core.detector import detect_model_type
        result = detect_model_type(str(tmp_path))
        assert self._model_type(result) == "fish-speech"

    def test_detect_result_has_confidence(self, tmp_path):
        """检测结果应包含 confidence 字段"""
        config = {"model_type": "cosyvoice"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        from core.detector import detect_model_type
        result = detect_model_type(str(tmp_path))
        if isinstance(result, dict):
            assert "confidence" in result

    def test_detect_result_has_method(self, tmp_path):
        """检测结果应包含 method 字段"""
        config = {"model_type": "xtts"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        from core.detector import detect_model_type
        result = detect_model_type(str(tmp_path))
        if isinstance(result, dict):
            assert "method" in result
