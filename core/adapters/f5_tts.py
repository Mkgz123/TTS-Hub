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
            import torch
            # Fallback: 检查模型文件是否存在
            model_files = list(self._model_path.glob("*.pt")) + list(self._model_path.glob("*.safetensors"))
            if not model_files:
                raise FileNotFoundError(f"未找到模型文件: {model_path}")
            self._model = None
            self._needs_package = True

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        if not self.is_loaded and not hasattr(self, "_needs_package"):
            raise RuntimeError("模型未加载")

        if hasattr(self, "_needs_package"):
            raise NotImplementedError(
                "F5-TTS 需要安装: pip install f5-tts\n"
                "或克隆仓库: git clone https://github.com/SWivid/F5-TTS"
            )

        ref_audio = request.extra.get("ref_audio", "")
        ref_text = request.extra.get("ref_text", "")

        audio, sr = self._model.infer(
            ref_file=ref_audio if ref_audio else None,
            ref_text=ref_text if ref_text else "",
            gen_text=request.text,
            speed=request.speed,
        )
        return TTSResponse.from_numpy(np.array(audio, dtype=np.float32), sr)

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": True,
            "emotion": False,
            "reference_audio": True,
        }
