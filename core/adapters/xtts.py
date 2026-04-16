"""XTTSv2 适配器 (Coqui TTS)"""

import numpy as np
from pathlib import Path
from typing import Optional
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


class XTTSAdapter(BaseTTSAdapter):
    model_type = "xtts"
    display_name = "XTTSv2 (Coqui)"
    supported_languages = ["zh", "en", "ja", "ko", "fr", "de", "es", "pt", "it"]
    default_sample_rate = 24000

    def __init__(self):
        super().__init__()
        self._tts_class = None

    def _import_tts(self):
        """尝试多个 import 路径导入 TTS"""
        import_errors = []
        
        # 尝试从 TTS 包导入
        try:
            from TTS.api import TTS
            self._tts_class = TTS
            return
        except ImportError as e:
            import_errors.append(f"TTS.api: {e}")
        
        # 尝试从 coqui_tts 导入
        try:
            from coqui_tts import TTS
            self._tts_class = TTS
            return
        except ImportError as e:
            import_errors.append(f"coqui_tts: {e}")
        
        # 尝试从 TTS.api 导入（不同版本）
        try:
            import TTS
            if hasattr(TTS, 'api'):
                from TTS.api import TTS as TTS_api
                self._tts_class = TTS_api
                return
            else:
                self._tts_class = TTS.TTS
                return
        except (ImportError, AttributeError) as e:
            import_errors.append(f"TTS: {e}")
        
        # 所有导入都失败
        error_msg = "\n".join(import_errors)
        raise ImportError(
            f"无法导入 TTS。请安装 Coqui TTS：\n"
            f"pip install TTS\n"
            f"参考: https://github.com/coqui-ai/TTS\n\n"
            f"导入错误详情:\n{error_msg}"
        )

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        """加载 XTTSv2 模型
        
        Args:
            model_path: 模型路径或 HuggingFace 模型名称
            device: 运行设备 ("cuda" 或 "cpu")
        """
        self._device = device
        self._import_tts()
        
        path = Path(model_path)
        
        # 如果是本地路径，检查是否存在
        if path.exists():
            # 检查是否是 XTTSv2 模型目录
            config_file = path / "config.json"
            model_file = path / "model.pth"
            
            if not (config_file.exists() and model_file.exists()):
                # 可能是其他格式的模型
                pass
        
        try:
            # 加载模型
            self._model = self._tts_class(model_path=model_path).to(device)
            
            # 验证模型是否支持 XTTSv2 功能
            if hasattr(self._model, 'is_multi_lingual') and not self._model.is_multi_lingual:
                print("警告: 加载的模型可能不是多语言模型")
            
        except Exception as e:
            raise RuntimeError(f"加载 XTTSv2 模型失败: {e}")

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """合成语音
        
        XTTSv2 支持：
        1. 多语言合成
        2. 声音克隆 (通过参考音频)
        """
        self._check_loaded()
        
        # 准备参数
        kwargs = {
            "text": request.text,
            "language": request.language,
        }
        
        # 处理参考音频
        reference_audio = None
        if request.speaker:
            # speaker 作为参考音频路径
            reference_audio = request.speaker
        elif request.extra.get("speaker_wav"):
            # 从 extra 参数获取参考音频
            reference_audio = request.extra["speaker_wav"]
        
        if reference_audio:
            # 验证参考音频文件是否存在
            if isinstance(reference_audio, str):
                ref_path = Path(reference_audio)
                if not ref_path.exists():
                    raise FileNotFoundError(f"参考音频文件不存在: {reference_audio}")
                kwargs["speaker_wav"] = reference_audio
            else:
                # 假设是音频数据
                kwargs["speaker_wav"] = reference_audio
        
        # 处理语速（XTTSv2 不直接支持，但可以通过 extra 参数传递）
        speed = request.speed
        if speed != 1.0:
            # XTTSv2 不直接支持语速调节，但可以通过后处理实现
            # 这里记录下来，后续可以后处理
            pass
        
        try:
            # 合成音频
            audio = self._model.tts(**kwargs)
            
            # 转换为 numpy 数组，确保是 float32
            if isinstance(audio, list):
                audio = np.array(audio, dtype=np.float32)
            elif hasattr(audio, 'numpy'):
                audio = audio.numpy().astype(np.float32)
            else:
                audio = np.array(audio, dtype=np.float32)
            
            # 应用语速调整（如果需要）
            if speed != 1.0:
                # 简单的重采样方法调整语速
                # 更精确的方法需要使用 librosa 或其他音频处理库
                target_length = int(len(audio) / speed)
                indices = np.linspace(0, len(audio) - 1, target_length)
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
            
            return TTSResponse.from_numpy(audio, self.default_sample_rate)
            
        except Exception as e:
            raise RuntimeError(f"合成失败: {e}")

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,  # 支持声音克隆
            "language": True,  # 支持多语言
            "speed": True,     # 通过后处理支持
            "emotion": False,  # XTTSv2 不直接支持情感控制
            "reference_audio": True,  # 支持参考音频
        }

    def get_speakers(self) -> list:
        """获取可用的说话人列表
        
        XTTSv2 通常通过参考音频指定声音，不提供预设说话人列表
        """
        return []  # XTTSv2 依赖参考音频，没有预设说话人