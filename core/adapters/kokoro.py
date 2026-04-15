"""Kokoro TTS 适配器"""

import numpy as np
from pathlib import Path
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
    supported_languages = ["en"]
    default_sample_rate = 24000

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        self._device = device
        path = Path(model_path)

        # 优先尝试 ONNX 版本（更轻量）
        onnx_files = list(path.glob("*.onnx"))
        if onnx_files:
            try:
                import kokoro_onnx
                self._model = kokoro_onnx.Kokoro(str(onnx_files[0]))
                self._onnx = True
                return
            except ImportError:
                pass

        # PyTorch 版本
        try:
            from kokoro import KPipeline
            self._model = KPipeline(lang_code="a", repo_id=str(path))
            self._onnx = False
        except ImportError:
            self._needs_package = True
            self._model_path = model_path

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        if hasattr(self, "_needs_package"):
            raise NotImplementedError(
                "Kokoro 需要安装: pip install kokoro-onnx\n"
                "或: pip install kokoro"
            )

        voice = request.speaker or "af_heart"
        speed = request.speed

        if getattr(self, "_onnx", False):
            audio, sr = self._model.create(text=request.text, voice=voice, speed=speed)
        else:
            generator = self._model(request.text, voice=voice, speed=speed)
            chunks = [a for _, _, a in generator]
            audio = np.concatenate(chunks)
            sr = 24000

        return TTSResponse.from_numpy(np.array(audio, dtype=np.float32), sr)

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": False,
            "speed": True,
            "emotion": False,
            "reference_audio": False,
        }

    def get_speakers(self) -> list:
        return KOKORO_VOICES
