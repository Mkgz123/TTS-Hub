"""ChatTTS 适配器"""

import numpy as np
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


class ChatTTSAdapter(BaseTTSAdapter):
    model_type = "chattts"
    display_name = "ChatTTS"
    supported_languages = ["zh"]
    default_sample_rate = 24000

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        self._device = device
        try:
            import ChatTTS
            chat = ChatTTS.Chat()
            chat.load(source="local", local_path=model_path, device=device)
            self._model = chat
        except ImportError:
            self._needs_package = True
            self._model_path = model_path

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        if hasattr(self, "_needs_package"):
            raise NotImplementedError(
                "ChatTTS 需要安装: pip install ChatTTS\n"
                "或克隆仓库: git clone https://github.com/2noise/ChatTTS"
            )

        import torch

        # ChatTTS 风格控制
        params_refine = {}
        if request.emotion:
            emotion_map = {
                "happy": "[laugh]",
                "sad": "[break_0]",
                "surprise": "[oral_2]",
            }
            if request.emotion in emotion_map:
                params_refine["prompt"] = f"[speed_{'fast' if request.speed > 1.2 else 'normal'}]{emotion_map[request.emotion]}"

        wavs = self._model.infer(
            [request.text],
            params_refine=params_refine,
            params_infer_code={
                "spk_emb": request.speaker if request.speaker else None,
                "temperature": 0.3,
                "top_P": 0.7,
            },
        )
        audio = np.array(wavs[0], dtype=np.float32)
        return TTSResponse.from_numpy(audio, self.default_sample_rate)

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": False,
            "speed": False,
            "emotion": True,
            "reference_audio": False,
        }
