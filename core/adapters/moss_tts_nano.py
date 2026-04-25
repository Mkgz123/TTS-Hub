"""MOSS-TTS-Nano 适配器

轻量级多语言 TTS 模型，仅 0.1B 参数，支持 CPU 实时推理。
零样本声音克隆，需要提供参考音频。

模型: OpenMOSS-Team/MOSS-TTS-Nano-100M
仓库: https://github.com/OpenMOSS/MOSS-TTS-Nano

Nano 没有独立的 AutoProcessor，需要分别加载：
- AutoTokenizer → 文本分词
- AutoModel (主模型) → 推理生成
- AutoModel (Audio Tokenizer) → 音频编解码 (MOSS-Audio-Tokenizer-Nano)
"""

import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
DEFAULT_AUDIO_TOKENIZER_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"


class MossTTSNanoAdapter(BaseTTSAdapter):
    model_type = "moss-tts-nano"
    display_name = "MOSS-TTS Nano (100M)"
    supported_languages = ["zh", "en", "ja", "ko", "fr", "de", "es"]
    default_sample_rate = 48000

    def __init__(self):
        super().__init__()
        self._tokenizer = None
        self._audio_tokenizer = None

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        self._device = device
        path = Path(model_path)

        # 自动检测 CUDA
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
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "MOSS-TTS-Nano 需要安装:\n"
                "  pip install transformers>=4.45 torch>=2.0 torchaudio\n"
                "或在 WebUI 环境管理中选择 moss-tts-nano 重新创建环境"
            )

        # 确定模型路径
        local_path = None
        if path.exists() and (path / "config.json").exists():
            model_id = str(path)
            local_path = path
        else:
            model_id = model_path if "/" in model_path else DEFAULT_MODEL_ID

        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # 1. 加载文本分词器（Nano 使用 SentencePiece tokenizer，非 processor）
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # 2. 加载主模型
        self._model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        self._model.eval()

        # 3. 加载音频编解码器（独立的 Nano 版本，比 TTSD 的 audio tokenizer 更小）
        audio_tokenizer_id = (
            getattr(self._model.config, "audio_tokenizer_pretrained_name_or_path", None)
            or DEFAULT_AUDIO_TOKENIZER_ID
        )

        # 优先使用本地下载的关联仓库
        if local_path is not None:
            codec_dir = local_path / "audio_tokenizer_nano"
            if (codec_dir / "config.json").exists():
                audio_tokenizer_id = str(codec_dir)

        self._audio_tokenizer = AutoModel.from_pretrained(
            audio_tokenizer_id,
            trust_remote_code=True,
        )
        if hasattr(self._audio_tokenizer, "to"):
            self._audio_tokenizer = self._audio_tokenizer.to(device)
        if hasattr(self._audio_tokenizer, "eval"):
            self._audio_tokenizer.eval()

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        self._check_loaded()

        import torch
        import numpy as np
        import soundfile as sf

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

        # 读取并预处理参考音频（使用 soundfile 避免 torchaudio 2.8+ 的 torchcodec 依赖）
        target_sr = self._get_audio_tokenizer_sampling_rate()
        data, sr = sf.read(str(ref_path), dtype="float32")
        wav = torch.from_numpy(data).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # (T,) → (1, T)
        elif wav.ndim == 2:
            wav = wav.transpose(0, 1)  # (T, C) → (C, T)
        if sr != target_sr:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, target_sr)

        # 编码参考音频
        wav = wav.to(self._device)
        encoded = self._audio_tokenizer.batch_encode([wav])
        prompt_audio_codes = self._model._normalize_audio_codes(encoded).to(self._device)

        # 构建推理输入
        text = request.text.strip()
        input_ids, attention_mask = self._model.build_inference_input_ids(
            text=text,
            text_tokenizer=self._tokenizer,
            mode="voice_clone",
            prompt_audio_codes=prompt_audio_codes,
            device=self._device,
        )

        # 生成
        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # 提取生成的音频 token IDs
        audio_token_ids = output.audio_token_ids

        # 解码为波形
        decoded = self._audio_tokenizer.batch_decode(
            [audio_token_ids],
            num_quantizers=self._model.config.n_vq,
        )

        # 提取波形和采样率
        audio_tensor, out_sr = self._model._extract_waveform_and_sample_rate(
            decoded, fallback_sample_rate=target_sr
        )

        audio = audio_tensor.detach().cpu().numpy()
        return TTSResponse.from_numpy(audio, out_sr)

    def _get_audio_tokenizer_sampling_rate(self) -> int:
        """从 audio tokenizer 获取目标采样率"""
        return self._model._resolve_audio_tokenizer_sample_rate(self._audio_tokenizer)

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": False,
            "emotion": False,
            "reference_audio": True,
        }
