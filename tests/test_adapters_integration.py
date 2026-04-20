"""
适配器集成测试
为每个注册的适配器验证接口合规性、元数据完整性和错误处理
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


# ============================================================
# 测试组 1：元数据与类属性
# ============================================================

class TestAdapterMetadata:
    """验证每个适配器的类属性声明完整"""

    def test_is_subclass_of_base(self, adapter):
        """适配器必须继承 BaseTTSAdapter"""
        assert isinstance(adapter, BaseTTSAdapter)

    def test_model_type_set(self, adapter):
        """model_type 必须为非空字符串"""
        assert isinstance(adapter.model_type, str)
        assert len(adapter.model_type) > 0

    def test_display_name_set(self, adapter):
        """display_name 必须为非空字符串"""
        assert isinstance(adapter.display_name, str)
        assert len(adapter.display_name) > 0

    def test_supported_languages_valid(self, adapter):
        """supported_languages 必须是合法语言代码列表"""
        assert isinstance(adapter.supported_languages, list)
        assert len(adapter.supported_languages) > 0
        valid_codes = {"zh", "en", "ja", "ko", "fr", "de", "es", "pt", "it", "ru", "ar"}
        for lang in adapter.supported_languages:
            assert lang in valid_codes, f"未知语言代码: {lang}"

    def test_default_sample_rate_reasonable(self, adapter):
        """default_sample_rate 必须在合理范围内"""
        sr = adapter.default_sample_rate
        assert isinstance(sr, int)
        assert 8000 <= sr <= 96000, f"采样率 {sr} 超出合理范围"


# ============================================================
# 测试组 2：接口合规性
# ============================================================

class TestAdapterInterface:
    """验证适配器实现了所有必须的接口方法"""

    def test_has_load_model(self, adapter):
        """必须实现 load_model"""
        assert callable(getattr(adapter, "load_model", None))

    def test_has_synthesize(self, adapter):
        """必须实现 synthesize"""
        assert callable(getattr(adapter, "synthesize", None))

    def test_has_unload(self, adapter):
        """必须实现 unload（继承即可）"""
        assert callable(getattr(adapter, "unload", None))

    def test_has_get_supported_features(self, adapter):
        """必须实现 get_supported_features"""
        assert callable(getattr(adapter, "get_supported_features", None))

    def test_has_get_speakers(self, adapter):
        """必须实现 get_speakers"""
        assert callable(getattr(adapter, "get_speakers", None))

    def test_has_validate_request(self, adapter):
        """必须实现 validate_request"""
        assert callable(getattr(adapter, "validate_request", None))


# ============================================================
# 测试组 3：get_supported_features() 返回值结构
# ============================================================

class TestSupportedFeatures:
    """验证 get_supported_features 返回正确的字典结构"""

    EXPECTED_KEYS = {"speaker", "language", "speed", "emotion", "reference_audio"}

    def test_returns_dict(self, adapter):
        features = adapter.get_supported_features()
        assert isinstance(features, dict)

    def test_has_all_required_keys(self, adapter):
        features = adapter.get_supported_features()
        for key in self.EXPECTED_KEYS:
            assert key in features, f"缺少功能键: {key}"

    def test_values_are_bool(self, adapter):
        features = adapter.get_supported_features()
        for key, val in features.items():
            assert isinstance(val, bool), f"{key} 的值 {val} 不是 bool"


# ============================================================
# 测试组 4：get_speakers() 返回值
# ============================================================

class TestGetSpeakers:
    """验证 get_speakers 返回合法列表"""

    def test_returns_list(self, adapter):
        try:
            speakers = adapter.get_speakers()
        except RuntimeError:
            # 部分适配器（如 CosyVoice）要求模型已加载才能返回说话人
            pytest.skip(f"{adapter.display_name} 需要加载模型才能获取说话人列表")
        assert isinstance(speakers, list)

    def test_speakers_are_strings(self, adapter):
        try:
            speakers = adapter.get_speakers()
        except RuntimeError:
            pytest.skip(f"{adapter.display_name} 需要加载模型才能获取说话人列表")
        for spk in speakers:
            assert isinstance(spk, str), f"说话人 {spk} 不是字符串"


# ============================================================
# 测试组 5：validate_request() 请求校验
# ============================================================

class TestValidateRequest:
    """验证请求校验逻辑"""

    def test_valid_request_passes(self, adapter, valid_request):
        result = adapter.validate_request(valid_request)
        assert result is None, f"合法请求被拒绝: {result}"

    def test_empty_text_rejected(self, adapter, empty_request):
        result = adapter.validate_request(empty_request)
        assert result is not None, "空文本请求应该被拒绝"

    def test_whitespace_only_rejected(self, adapter):
        req = TTSRequest(text="   \n\t  ")
        result = adapter.validate_request(req)
        assert result is not None, "纯空白请求应该被拒绝"

    def test_long_text_rejected(self, adapter, long_text_request):
        result = adapter.validate_request(long_text_request)
        assert result is not None, "超长文本应该被拒绝"


# ============================================================
# 测试组 6：未加载模型时的错误处理
# ============================================================

class TestUnloadedBehavior:
    """验证模型未加载时的正确错误处理"""

    def test_initial_not_loaded(self, adapter):
        """新实例应该是未加载状态"""
        assert not adapter.is_loaded

    def test_synthesize_raises_when_unloaded(self, adapter, valid_request):
        """未加载时调用 synthesize 应抛出异常"""
        with pytest.raises((RuntimeError, NotImplementedError)):
            adapter.synthesize(valid_request)

    def test_unload_on_unloaded_is_safe(self, adapter):
        """对未加载的适配器调用 unload 不应崩溃"""
        adapter.unload()  # 应静默完成


# ============================================================
# 测试组 7：load_model() 错误处理（无实际模型）
# ============================================================

class TestLoadModelGracefulFailure:
    """验证 load_model 在无模型/无依赖时的优雅降行"""

    def test_load_nonexistent_graceful(self, adapter):
        """加载不存在的路径应优雅处理（静默设置 _needs_package 或抛异常）"""
        try:
            adapter.load_model("/nonexistent/model/path", device="cpu")
        except (RuntimeError, ImportError, FileNotFoundError, NotImplementedError):
            pass  # 抛异常是可接受的行为
        # 静默设置 _needs_package 也是可接受的
        assert not adapter.is_loaded, "加载失败后不应处于已加载状态"

    def test_load_empty_dir_graceful(self, adapter, tmp_path):
        """加载空目录应优雅处理"""
        try:
            adapter.load_model(str(tmp_path), device="cpu")
        except (RuntimeError, ImportError, FileNotFoundError, NotImplementedError):
            pass
        assert not adapter.is_loaded, "加载失败后不应处于已加载状态"


# ============================================================
# 测试组 8：TTSResponse 数据完整性
# ============================================================

class TestTTSResponse:
    """验证 TTSResponse 工厂方法和属性"""

    def test_from_numpy_basic(self):
        audio = np.zeros(24000, dtype=np.float32)
        resp = TTSResponse.from_numpy(audio, 24000)
        assert resp.sample_rate == 24000
        assert resp.duration == 1.0
        assert len(resp.audio) == 24000

    def test_from_numpy_half_second(self):
        audio = np.zeros(12000, dtype=np.float32)
        resp = TTSResponse.from_numpy(audio, 24000)
        assert resp.duration == 0.5

    def test_audio_is_float32(self):
        audio = np.zeros(48000, dtype=np.float32)
        resp = TTSResponse.from_numpy(audio, 48000)
        assert resp.audio.dtype == np.float32

    def test_audio_from_int_converts(self):
        """非 float32 输入也能工作（numpy 会自动处理）"""
        audio = np.zeros(16000, dtype=np.float64)
        resp = TTSResponse.from_numpy(audio, 16000)
        assert resp.duration == 1.0


# ============================================================
# 测试组 9：TTSRequest 数据类
# ============================================================

class TestTTSRequest:
    """验证 TTSRequest 默认值和字段"""

    def test_minimal_request(self):
        req = TTSRequest(text="hello")
        assert req.text == "hello"
        assert req.speaker is None
        assert req.language == "zh"
        assert req.speed == 1.0
        assert req.emotion is None
        assert req.sample_rate == 24000
        assert req.extra == {}

    def test_full_request(self):
        req = TTSRequest(
            text="测试",
            speaker="alice",
            language="en",
            speed=1.5,
            emotion="happy",
            sample_rate=16000,
            extra={"key": "value"},
        )
        assert req.speaker == "alice"
        assert req.language == "en"
        assert req.speed == 1.5
        assert req.emotion == "happy"
        assert req.sample_rate == 16000
        assert req.extra == {"key": "value"}


# ============================================================
# 测试组 10：注册表一致性
# ============================================================

class TestRegistryConsistency:
    """验证注册表与实际适配器的一致性"""

    def test_all_registered_adapters_importable(self):
        """所有注册的适配器都能成功 import（不要求模型依赖）"""
        from core.registry import _ADAPTER_REGISTRY
        import importlib

        for model_type, (module_path, class_name) in _ADAPTER_REGISTRY.items():
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            assert issubclass(cls, BaseTTSAdapter), \
                f"{model_type}: {class_name} 不是 BaseTTSAdapter 的子类"

    def test_adapter_model_type_matches_registry(self):
        """每个适配器的 model_type 应与注册表 key 一致"""
        from core.registry import _ADAPTER_REGISTRY, _adapter_cache
        _adapter_cache.clear()

        for model_type in _ADAPTER_REGISTRY:
            from core.registry import get_adapter
            adapter = get_adapter(model_type)
            if adapter is not None:
                assert adapter.model_type == model_type, \
                    f"注册为 '{model_type}' 但适配器声明为 '{adapter.model_type}'"
