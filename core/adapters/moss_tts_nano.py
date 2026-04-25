"""MOSS-TTS-Nano 适配器

轻量级多语言 TTS 模型，仅 0.1B 参数，支持 CPU 实时推理。
零样本声音克隆，需要提供参考音频。

模型: OpenMOSS-Team/MOSS-TTS-Nano-100M
仓库: https://github.com/OpenMOSS/MOSS-TTS-Nano
"""

import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Nano-100M"


class MossTTSNanoAdapter(BaseTTSAdapter):
    model_type = "moss-tts-nano"
    display_name = "MOSS-TTS Nano (100M)"
    supported_languages = ["zh", "en", "ja", "ko", "fr", "de", "es"]
    default_sample_rate = 48000

    def __init__(self):
        super().__init__()
        self._processor = None

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        self._device = device
        path = Path(model_path)

        # 自动检测 CUDA（Nano 在 CPU 上也能流畅运行）
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    device = "cpu"
            except ImportError:
                device = "cpu"
        self._device = device

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise ImportError(
                "MOSS-TTS-Nano 需要安装:\n"
                "  pip install transformers>=4.45 torch>=2.0 torchaudio\n"
                "或在 WebUI 环境管理中选择 moss-tts-nano 重新创建环境"
            )

        # 确定模型路径
        if path.exists() and (path / "config.json").exists():
            model_id = str(path)
        else:
            model_id = model_path if "/" in model_path else DEFAULT_MODEL_ID

        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # Nano 的 processor 没有独立的 audio_tokenizer，codec 内嵌在 processor 中
        self._model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        self._model.eval()

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        self._check_loaded()

        import torch
        import torchaudio

        # 获取参考音频
        ref_audio = request.speaker or request.extra.get("ref_audio", "")
        if not ref_audio:
            raise ValueError(
                "MOSS-TTS-Nano 需要参考音频进行声音克隆。\n"
                "请通过 speaker 参数或 extra['ref_audio'] 传入参考音频路径。"
            )

        ref_path = Path(ref_audio)
        if not ref_path.exists():
            raise FileNotFoundError(f"参考音频不存在: {ref_audio}")

        # 读取并预处理参考音频
        target_sr = int(self._processor.model_config.sampling_rate)
        wav, sr = torchaudio.load(str(ref_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)

        # 编码参考音频
        reference_audio_codes = self._processor.encode_audios_from_wav(
            [wav], sampling_rate=target_sr
        )
        prompt_audio_code = reference_audio_codes[0]

        # 参考文本和目标文本
        ref_text = request.extra.get("ref_text", "[S1]这是参考音频。")

        text = request.text.strip()
        if not text.startswith("["):
            text = f"[S1]{text}"

        full_text = f"{ref_text} {text}"

        # 构建对话（单说话人 continuation 模式）
        conversations = [
            [
                self._processor.build_user_message(
                    text=full_text,
                    reference=reference_audio_codes,
                ),
                self._processor.build_assistant_message(
                    audio_codes_list=[prompt_audio_code]
                ),
            ],
        ]

        # 推理
        batch = self._processor(conversations, mode="continuation")
        if self._device == "cuda":
            batch = {k: v.to(self._device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=2000,
            )

        messages = self._processor.decode(outputs)
        if not messages:
            raise RuntimeError("合成失败：未生成音频数据")

        audio_tensor = messages[0].audio_codes_list[0]
        audio = audio_tensor.detach().cpu().to(torch.float32).numpy()

        return TTSResponse.from_numpy(audio, target_sr)

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": False,
            "emotion": False,
            "reference_audio": True,
        }
