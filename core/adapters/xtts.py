"""XTTSv2 适配器 (Coqui TTS)"""

import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


class XTTSAdapter(BaseTTSAdapter):
    model_type = "xtts"
    display_name = "XTTSv2 (Coqui)"
    supported_languages = ["zh", "en", "ja", "ko", "fr", "de", "es", "pt", "it"]
    default_sample_rate = 24000

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        self._device = device
        try:
            from TTS.api import TTS
            self._model = TTS(model_path=model_path).to(device)
        except ImportError:
            self._needs_package = True
            self._model_path = model_path

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        if hasattr(self, "_needs_package"):
            raise NotImplementedError(
                "XTTSv2 需要安装: pip install TTS\n"
                "参考: https://github.com/coqui-ai/TTS"
            )

        kwargs = {
            "text": request.text,
            "language": request.language,
        }

        if request.speaker:
            # speaker 是参考音频路径
            kwargs["speaker_wav"] = request.speaker
        elif request.extra.get("speaker_wav"):
            kwargs["speaker_wav"] = request.extra["speaker_wav"]

        audio = self._model.tts(**kwargs)
        return TTSResponse.from_numpy(np.array(audio, dtype=np.float32), self.default_sample_rate)

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": False,
            "emotion": False,
            "reference_audio": True,
        }
