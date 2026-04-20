"""共享 fixtures"""

import sys
import pytest
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse
from core.registry import _ADAPTER_REGISTRY, _adapter_cache, get_adapter


# ============================================================
# 所有已注册的适配器类型
# ============================================================
ALL_ADAPTER_TYPES = list(_ADAPTER_REGISTRY.keys())


@pytest.fixture(params=ALL_ADAPTER_TYPES, ids=ALL_ADAPTER_TYPES)
def adapter_type(request):
    """参数化 fixture：每个适配器类型都会运行一次"""
    return request.param


@pytest.fixture
def adapter(adapter_type):
    """获取适配器实例（不加载模型）"""
    # 清除缓存，确保每次测试拿到新实例
    _adapter_cache.pop(adapter_type, None)
    adapter = get_adapter(adapter_type)
    if adapter is None:
        pytest.skip(f"适配器 {adapter_type} 依赖未安装，跳过")
    return adapter


@pytest.fixture
def valid_request():
    """标准合法请求"""
    return TTSRequest(text="你好，这是一段测试文本。")


@pytest.fixture
def empty_request():
    """空文本请求"""
    return TTSRequest(text="")


@pytest.fixture
def long_text_request():
    """超长文本请求"""
    return TTSRequest(text="测试" * 3000)  # 6000 字符


@pytest.fixture
def request_with_speed():
    """带语速参数的请求"""
    return TTSRequest(text="测试语速", speed=1.5)


@pytest.fixture
def request_with_speaker():
    """带说话人参数的请求"""
    return TTSRequest(text="测试说话人", speaker="test_speaker")
