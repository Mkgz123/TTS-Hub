"""CosyVoice 适配器"""

import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


class CosyVoiceAdapter(BaseTTSAdapter):
    model_type = "cosyvoice"
    display_name = "CosyVoice"
    supported_languages = ["zh", "en"]
    default_sample_rate = 22050

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        self._device = device
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
            self._model = CosyVoice(model_path, device=device)
        except ImportError:
            self._needs_package = True
            self._model_path = model_path

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        if hasattr(self, "_needs_package"):
            raise NotImplementedError(
                "CosyVoice 需要克隆仓库: git clone https://github.com/FunAudioLLM/CosyVoice\n"
                "参考 README 安装依赖"
            )

        if request.extra.get("mode") == "zero_shot" and request.extra.get("prompt_audio"):
            # 零样本声音克隆
            results = self._model.inference_zero_shot(
                request.text,
                request.extra["prompt_audio"],
                request.extra.get("prompt_text", ""),
            )
        elif request.speaker:
            # 指定说话人
            results = self._model.inference_sft(request.text, request.speaker)
        else:
            # 默认
            results = self._model.inference_sft(request.text, "")

        audio = np.array(list(results)[0]["tts_speech"].numpy().flatten(), dtype=np.float32)
        return TTSResponse.from_numpy(audio, self.default_sample_rate)

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": True,
            "emotion": False,
            "reference_audio": True,
        }

    def get_speakers(self) -> list:
        if self.is_loaded and hasattr(self._model, "list_avaliable_spks"):
            return list(self._model.list_avaliable_spks())
        return []
