"""Kokoro TTS 适配器"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Any
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


# Kokoro 已知的 voices
KOKORO_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
]


class KokoroAdapter(BaseTTSAdapter):
    model_type = "kokoro"
    display_name = "Kokoro TTS"
    supported_languages = ["en"]  # 主要支持英语
    default_sample_rate = 24000

    def __init__(self):
        super().__init__()
        self._onnx = False
        self._kokoro_onnx_class = None
        self._kpipeline_class = None

    def _import_kokoro_onnx(self):
        """尝试导入 kokoro_onnx"""
        import_errors = []
        
        # 尝试标准导入
        try:
            import kokoro_onnx
            self._kokoro_onnx_class = kokoro_onnx.Kokoro
            return True
        except ImportError as e:
            import_errors.append(f"kokoro_onnx: {e}")
        
        # 尝试从本地路径导入
        try:
            import sys
            possible_paths = [
                Path.home() / "kokoro-onnx",
                Path("/opt/kokoro-onnx"),
                Path("./kokoro-onnx"),
            ]
            for p in possible_paths:
                if p.exists() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            
            import kokoro_onnx
            self._kokoro_onnx_class = kokoro_onnx.Kokoro
            return True
        except ImportError as e:
            import_errors.append(f"local kokoro_onnx: {e}")
        
        return False

    def _import_kokoro_pytorch(self):
        """尝试导入 PyTorch 版本的 Kokoro"""
        import_errors = []
        
        # 尝试从 kokoro 包导入
        try:
            from kokoro import KPipeline
            self._kpipeline_class = KPipeline
            return True
        except ImportError as e:
            import_errors.append(f"kokoro: {e}")
        
        # 尝试从 kokoro_tts 导入
        try:
            from kokoro_tts import KPipeline
            self._kpipeline_class = KPipeline
            return True
        except ImportError as e:
            import_errors.append(f"kokoro_tts: {e}")
        
        # 尝试从本地路径导入
        try:
            import sys
            possible_paths = [
                Path.home() / "kokoro",
                Path("/opt/kokoro"),
                Path("./kokoro"),
            ]
            for p in possible_paths:
                if p.exists() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            
            from kokoro import KPipeline
            self._kpipeline_class = KPipeline
            return True
        except ImportError as e:
            import_errors.append(f"local kokoro: {e}")
        
        return False

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        """加载 Kokoro 模型
        
        Args:
            model_path: 模型目录或文件路径
            device: 运行设备 ("cuda" 或 "cpu")
        """
        self._device = device
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 优先尝试 ONNX 版本（更轻量）
        onnx_files = list(path.glob("*.onnx")) if path.is_dir() else []
        if path.is_file() and path.suffix == ".onnx":
            onnx_files = [path]
        
        if onnx_files:
            if self._import_kokoro_onnx():
                try:
                    self._model = self._kokoro_onnx_class(str(onnx_files[0]))
                    self._onnx = True
                    return
                except Exception as e:
                    print(f"ONNX 加载失败，尝试 PyTorch 版本: {e}")
        
        # PyTorch 版本
        if self._import_kokoro_pytorch():
            try:
                # KPipeline 需要 lang_code 参数
                # repo_id 可以是本地路径或 HuggingFace 仓库 ID
                if path.is_dir():
                    # 如果是本地目录，尝试找到 config.json 或 model.bin
                    config_file = path / "config.json"
                    if config_file.exists():
                        self._model = self._kpipeline_class(lang_code="a", repo_id=str(path))
                    else:
                        # 假设目录名就是仓库 ID
                        self._model = self._kpipeline_class(lang_code="a", repo_id=path.name)
                else:
                    # 假设是 HuggingFace 仓库 ID
                    self._model = self._kpipeline_class(lang_code="a", repo_id=model_path)
                self._onnx = False
                return
            except Exception as e:
                raise RuntimeError(f"PyTorch Kokoro 加载失败: {e}")
        
        # 两种版本都无法导入
        raise ImportError(
            "无法导入 Kokoro。请安装以下依赖之一：\n"
            "1. ONNX 版本 (推荐): pip install kokoro-onnx\n"
            "2. PyTorch 版本: pip install kokoro\n"
            "参考: https://github.com/hexgrad/kokoro"
        )

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """合成语音"""
        self._check_loaded()
        
        voice = request.speaker or "af_heart"
        speed = request.speed
        
        # 验证 voice 是否有效
        if voice not in KOKORO_VOICES:
            # 不强制要求，因为可能有其他 voices
            pass
        
        try:
            if self._onnx:
                # ONNX 版本
                audio, sr = self._model.create(
                    text=request.text,
                    voice=voice,
                    speed=speed
                )
                # 确保音频是 float32 numpy 数组
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)
                else:
                    audio = audio.astype(np.float32)
            else:
                # PyTorch 版本
                generator = self._model(
                    request.text,
                    voice=voice,
                    speed=speed
                )
                # generator 返回 (graphemes, phonemes, audio) 元组
                chunks = []
                for _, _, audio_chunk in generator:
                    if isinstance(audio_chunk, np.ndarray):
                        chunks.append(audio_chunk)
                    elif hasattr(audio_chunk, 'numpy'):
                        chunks.append(audio_chunk.numpy())
                    else:
                        chunks.append(np.array(audio_chunk))
                
                if not chunks:
                    raise RuntimeError("合成失败：未获取到音频数据")
                
                audio = np.concatenate(chunks)
                audio = audio.astype(np.float32)
                sr = self.default_sample_rate
            
            return TTSResponse.from_numpy(audio, sr)
            
        except Exception as e:
            raise RuntimeError(f"合成失败: {e}")

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": False,  # 主要支持英语
            "speed": True,
            "emotion": False,
            "reference_audio": False,
        }

    def get_speakers(self) -> list:
        """获取可用的说话人列表"""
        return KOKORO_VOICES.copy()