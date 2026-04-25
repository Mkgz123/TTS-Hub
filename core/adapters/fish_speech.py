"""Fish-Speech 适配器"""

import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


class FishSpeechAdapter(BaseTTSAdapter):
    model_type = "fish-speech"
    display_name = "Fish-Speech"
    supported_languages = ["zh", "en"]
    default_sample_rate = 44100

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        self._device = device
        self._model_path = Path(model_path)

        if not self._model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        try:
            from fish_speech.inference_engine import FishSpeechInferenceEngine
            self._model = FishSpeechInferenceEngine(
                model_path=str(self._model_path),
                device=device,
            )
            self._engine_type = "fish_speech"
            self._needs_package = False
        except ImportError:
            self._needs_package = True
        except Exception as e:
            raise RuntimeError(f"加载 Fish-Speech 模型失败: {e}")

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        if hasattr(self, "_needs_package"):
            raise NotImplementedError(
                "Fish-Speech 需要安装 fish_speech 包: pip install fish-speech\n"
                "或克隆仓库: git clone https://github.com/fishaudio/fish-speech"
            )

        self._check_loaded()

        audio = self._model.inference(
            text=request.text,
            speaker=request.speaker,
            language=request.language,
            speed=request.speed,
        )

        return TTSResponse.from_numpy(
            np.array(audio, dtype=np.float32),
            self.default_sample_rate,
        )

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": True,
            "emotion": False,
            "reference_audio": True,
        }

    def get_speakers(self) -> list:
        if self.is_loaded and hasattr(self._model, "list_speakers"):
            return list(self._model.list_speakers())
        return []