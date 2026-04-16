"""F5-TTS 适配器"""

import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


class F5TTSAdapter(BaseTTSAdapter):
    model_type = "f5-tts"
    display_name = "F5-TTS"
    supported_languages = ["zh", "en"]
    default_sample_rate = 24000

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        self._device = device
        self._model_path = Path(model_path)
        
        try:
            from f5_tts.api import F5TTS
            self._model = F5TTS(
                model=str(self._model_path),
                device=device,
            )
        except ImportError:
            self._needs_package = True
        except Exception as e:
            # 检查模型文件是否存在
            model_files = list(self._model_path.glob("*.pt")) + list(self._model_path.glob("*.safetensors"))
            if not model_files:
                raise FileNotFoundError(f"未找到模型文件: {model_path}")
            raise RuntimeError(f"加载 F5-TTS 模型失败: {e}")

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        if hasattr(self, "_needs_package"):
            raise NotImplementedError(
                "F5-TTS 需要安装: pip install f5-tts\n"
                "或克隆仓库: git clone https://github.com/SWivid/F5-TTS"
            )
        
        self._check_loaded()

        # 获取参考音频和文本
        ref_audio = request.extra.get("ref_audio", "")
        ref_text = request.extra.get("ref_text", "")
        
        # 如果提供了 speaker 且看起来像文件路径，用作参考音频
        if request.speaker and Path(request.speaker).exists():
            ref_audio = request.speaker

        # 调用 F5-TTS 推理
        # 注意: 返回值可能是 (audio, sr) 或 (audio, sr, spectogram)
        result = self._model.infer(
            ref_file=ref_audio if ref_audio else None,
            ref_text=ref_text if ref_text else "",
            gen_text=request.text,
            speed=request.speed,
        )
        
        # 处理不同版本的返回值
        if len(result) == 2:
            audio, sr = result
        elif len(result) == 3:
            audio, sr, _ = result
        else:
            raise RuntimeError(f"Unexpected return value from F5-TTS infer: {len(result)} values")

        return TTSResponse.from_numpy(np.array(audio, dtype=np.float32), sr)

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": True,
            "emotion": False,
            "reference_audio": True,
        }