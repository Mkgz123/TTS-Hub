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
        try:
            from fish_speech.inference import FishSpeechEngine
            self._model = FishSpeechEngine(
                model_path=model_path,
                device=device,
            )
        except ImportError:
            # Fallback: 直接用 transformers + torch 加载
            import torch
            from transformers import Qwen2ForCausalLM, AutoTokenizer
            path = Path(model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
            self._model = Qwen2ForCausalLM.from_pretrained(
                str(path),
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
            )
            self._custom_load = True

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        self._check_loaded()

        if hasattr(self, "_custom_load"):
            # Fallback 模式 — 基础推理（实际效果需要 fish_speech 库）
            raise NotImplementedError(
                "Fish-Speech 需要安装 fish_speech 包: pip install fish-speech\n"
                "或克隆仓库: git clone https://github.com/fishaudio/fish-speech"
            )

        audio = self._model.synthesize(
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
            "speed": False,
            "emotion": False,
            "reference_audio": True,
        }

    def _check_loaded(self):
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")
