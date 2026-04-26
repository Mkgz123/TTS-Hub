"""MOSS-SoundEffect 适配器

文字描述生成音效模型 — 用自然语言描述环境音效，直接生成高质量音效。
模型接收 ambient_sound（音效描述），生成对应音频。

模型: OpenMOSS-Team/MOSS-SoundEffect (8B)
仓库: https://github.com/OpenMOSS/MOSS-TTS

注意: 仅支持 CUDA，CPU 不可用（8B 参数）。
"""

import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-SoundEffect"


class MossSoundEffectAdapter(BaseTTSAdapter):
    model_type = "moss-soundeffect"
    display_name = "MOSS-SoundEffect"
    supported_languages = ["zh", "en"]
    default_sample_rate = 24000

    def __init__(self):
        super().__init__()
        self._processor = None

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise ImportError(
                "MOSS-SoundEffect 需要安装:\n"
                "  pip install transformers>=5.0 torch>=2.5.1 torchaudio soundfile\n"
                "或在 WebUI 环境管理中选择 moss-soundeffect 重新创建环境"
            )

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "MOSS-SoundEffect (8B) 需要 CUDA GPU。\n"
                "当前系统未检测到 NVIDIA GPU，无法运行此模型。"
            )
        self._device = device

        path = Path(model_path)
        if path.exists() and (path / "config.json").exists():
            model_id = str(path)
        else:
            model_id = model_path if "/" in model_path else DEFAULT_MODEL_ID

        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # CUDA SDPA 后端
        if device == "cuda":
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        if hasattr(self._processor, "audio_tokenizer"):
            self._processor.audio_tokenizer = self._processor.audio_tokenizer.to(device)
        if hasattr(self._processor.audio_tokenizer, "eval"):
            self._processor.audio_tokenizer.eval()

        self._model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=dtype,
        ).to(device)
        self._model.eval()

        self._sample_rate = int(
            getattr(self._processor.model_config, "sampling_rate", 24000)
        )

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        self._check_loaded()

        import torch

        text = (request.text or "").strip()
        if not text:
            raise ValueError("请输入音效描述。\n例如: 雨声淅沥、远处雷声隆隆。")

        # 采样参数（SoundEffect 推荐默认值）
        audio_temperature = float(request.extra.get("audio_temperature", 1.5))
        audio_top_p = float(request.extra.get("audio_top_p", 0.6))
        audio_top_k = int(request.extra.get("audio_top_k", 50))
        audio_repetition_penalty = float(request.extra.get("audio_repetition_penalty", 1.2))
        max_new_tokens = int(request.extra.get("max_new_tokens", 4096))

        # 构建对话 — SoundEffect 使用 ambient_sound 参数
        conversations = [[
            self._processor.build_user_message(ambient_sound=text)
        ]]

        batch = self._processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
            )

        messages = self._processor.decode(outputs)
        if not messages or messages[0] is None:
            raise RuntimeError("模型未返回可解码的音频结果。")

        audio_tensor = messages[0].audio_codes_list[0]
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.as_tensor(audio_tensor, dtype=torch.float32)

        audio = audio_tensor.detach().cpu().to(torch.float32).numpy()
        if audio.ndim > 1:
            audio = audio.reshape(-1)

        return TTSResponse.from_numpy(audio.astype(np.float32), self._sample_rate)

    def get_supported_features(self) -> dict:
        return {
            "speaker": False,
            "language": False,
            "speed": False,
            "emotion": False,
            "reference_audio": False,
            "instruction": False,
            "ambient_sound": True,  # 音效描述
        }
