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
        self._model_path = model_path
        
        try:
            import ChatTTS
            chat = ChatTTS.Chat()
            chat.load(
                source="local",
                local_path=model_path,
                device=device
            )
            self._model = chat
            self._needs_package = False
        except ImportError:
            self._needs_package = True
        except Exception as e:
            raise RuntimeError(f"加载 ChatTTS 模型失败: {e}")

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        if hasattr(self, "_needs_package"):
            raise NotImplementedError(
                "ChatTTS 需要安装: pip install ChatTTS\n"
                "或克隆仓库: git clone https://github.com/2noise/ChatTTS"
            )
        
        self._check_loaded()

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

        # 构建推理参数
        params_infer_code = {
            "temperature": 0.3,
            "top_P": 0.7,
        }
        
        # 如果提供了说话人（可以是说话人嵌入或路径）
        if request.speaker:
            params_infer_code["spk_emb"] = request.speaker
        
        # 处理语速
        if request.speed != 1.0:
            # ChatTTS 通过 prompt 控制语速
            speed_prompt = f"[speed_{'fast' if request.speed > 1.0 else 'slow'}]"
            if "prompt" in params_refine:
                params_refine["prompt"] = speed_prompt + params_refine["prompt"]
            else:
                params_refine["prompt"] = speed_prompt

        # 调用 ChatTTS 推理
        wavs = self._model.infer(
            [request.text],
            params_refine=params_refine,
            params_infer_code=params_infer_code,
        )
        
        # wavs 是一个列表，每个元素是一个音频数组
        if isinstance(wavs, list) and len(wavs) > 0:
            audio = wavs[0]
        else:
            audio = wavs
        
        # 确保输出为 float32
        audio_array = np.array(audio, dtype=np.float32)
        
        return TTSResponse.from_numpy(audio_array, self.default_sample_rate)

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": False,
            "speed": True,  # ChatTTS 通过 prompt 控制语速
            "emotion": True,
            "reference_audio": False,
        }