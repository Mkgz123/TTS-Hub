"""
统一 TTS 推理接口抽象基类
所有模型适配器继承此类，实现统一的 load / synthesize / unload 接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class TTSRequest:
    """统一的 TTS 请求"""
    text: str
    speaker: Optional[str] = None       # 说话人 ID 或参考音频路径
    language: str = "zh"                 # 语言代码
    speed: float = 1.0                   # 语速倍率
    emotion: Optional[str] = None        # 情感（仅部分模型支持）
    sample_rate: int = 24000             # 期望输出采样率
    extra: dict = field(default_factory=dict)  # 模型特有参数


@dataclass
class TTSResponse:
    """统一的 TTS 响应"""
    audio: np.ndarray                    # 音频数据 (float32, -1.0 ~ 1.0)
    sample_rate: int                     # 实际采样率
    duration: float                      # 时长（秒）

    @classmethod
    def from_numpy(cls, audio: np.ndarray, sample_rate: int) -> "TTSResponse":
        duration = len(audio) / sample_rate
        return cls(audio=audio, sample_rate=sample_rate, duration=duration)


class BaseTTSAdapter(ABC):
    """TTS 适配器抽象基类"""

    # 子类必须声明
    model_type: str = ""                 # 模型类型标识 (如 "fish-speech")
    display_name: str = ""               # 显示名称
    supported_languages: list = []       # 支持的语言 ["zh", "en"]
    default_sample_rate: int = 24000     # 默认采样率

    def __init__(self):
        self._model = None
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @abstractmethod
    def load_model(self, model_path: str, device: str = "cuda") -> None:
        """加载模型

        Args:
            model_path: 模型目录或文件路径
            device: "cuda" 或 "cpu"
        """
        ...

    @abstractmethod
    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """合成语音"""
        ...

    def unload(self) -> None:
        """卸载模型，释放显存"""
        self._model = None
        if self._device == "cuda":
            import torch
            torch.cuda.empty_cache()

    def get_supported_features(self) -> dict:
        """返回支持的功能列表（子类可覆盖）"""
        return {
            "speaker": False,
            "language": False,
            "speed": False,
            "emotion": False,
            "reference_audio": False,
        }

    def get_speakers(self) -> list:
        """返回可用说话人列表（子类可覆盖）"""
        return []

    def _check_loaded(self):
        """通用加载检查（子类可直接调用）"""
        if not self.is_loaded:
            raise RuntimeError(f"{self.display_name} 模型未加载，请先调用 load_model()")

    def validate_request(self, request: TTSRequest) -> Optional[str]:
        """校验请求，返回错误信息或 None"""
        if not request.text or not request.text.strip():
            return "文本不能为空"
        if len(request.text) > 5000:
            return "文本过长（最多 5000 字符）"
