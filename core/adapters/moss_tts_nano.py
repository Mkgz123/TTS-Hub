"""MOSS-TTS-Nano 适配器

轻量级多语言 TTS 模型，仅 0.1B 参数，支持 CPU 实时推理。
零样本声音克隆，需要提供参考音频。

模型: OpenMOSS-Team/MOSS-TTS-Nano-100M
仓库: https://github.com/OpenMOSS/MOSS-TTS-Nano

注意: 必须使用 torch<2.8，因为 torchaudio 2.8+ 的 load/save 需要 torchcodec
+ FFmpeg 共享库，在 Windows 上不可行。使用 soundfile 加载参考音频。
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
        self._audio_tokenizer = None

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        self._device = device
        path = Path(model_path)

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
                "  pip install transformers==4.57.1 torch==2.7.0 torchaudio==2.7.0\n"
                "  pip install sentencepiece soundfile\n"
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
        attn = "sdpa" if device == "cuda" else "eager"

        # 1. 文本分词器
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # 2. 主模型
        # Nano 的 config.json 写死了 flash_attention_2，但其 __init__ 直接读
        # config.attn_implementation 而非 config._attn_implementation，
        # 导致 from_pretrained(attn_implementation=...) 参数被忽略。
        # 这里直接修改 config 后再加载。
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.attn_implementation = attn
        config.local_transformer_attn_implementation = attn

        self._model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            config=config,
        ).to(device)
        self._model.eval()

        # 3. 音频编解码器
        audio_tokenizer_id = (
            getattr(self._model.config, "audio_tokenizer_pretrained_name_or_path", None)
            or DEFAULT_AUDIO_TOKENIZER_ID
        )
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
        import soundfile as sf
        import torchaudio

        # ── 参考音频 ──
        ref_audio = request.speaker or request.extra.get("ref_audio", "")
        if not ref_audio:
            raise ValueError(
                "MOSS-TTS-Nano 需要参考音频进行声音克隆。\n"
                "请通过 speaker 参数或 extra['ref_audio'] 传入参考音频路径。"
            )
        ref_path = Path(ref_audio)
        if not ref_path.exists():
            raise FileNotFoundError(f"参考音频不存在: {ref_audio}")

        # ── Monkey-patch torchaudio I/O → soundfile ──
        _orig_load = torchaudio.load
        _orig_save = torchaudio.save

        def _sf_load(path, *args, **kwargs):
            data, sr = sf.read(str(path), dtype="float32")
            wav = torch.from_numpy(data).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.T
            return wav, sr

        def _sf_save(path, waveform, sample_rate, *args, **kwargs):
            audio = waveform.detach().cpu().numpy()
            if audio.ndim == 2:
                audio = audio.T
            sf.write(str(path), audio, int(sample_rate))

        torchaudio.load = _sf_load
        torchaudio.save = _sf_save

        try:
            import tempfile
            tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_out.close()

            result = self._model.inference(
                text=request.text.strip(),
                output_audio_path=tmp_out.name,
                mode="voice_clone",
                reference_audio_path=str(ref_path),
                text_tokenizer=self._tokenizer,
                audio_tokenizer=self._audio_tokenizer,
                device=self._device,
            )

            waveform = result["waveform"]
            if waveform is None:
                raise RuntimeError("inference 未返回波形数据")

            audio = waveform.detach().cpu().numpy()
            if audio.ndim == 3:
                audio = audio[0]
            if audio.ndim == 2:
                audio = audio.T
            return TTSResponse.from_numpy(audio, int(result["sample_rate"]))
        finally:
            torchaudio.load = _orig_load
            torchaudio.save = _orig_save

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": False,
            "emotion": False,
            "reference_audio": True,
        }
