"""MOSS-TTS-Nano 适配器

轻量级多语言 TTS 模型，仅 0.1B 参数，支持 CPU 实时推理。
零样本声音克隆，需要提供参考音频。

模型: OpenMOSS-Team/MOSS-TTS-Nano-100M
仓库: https://github.com/OpenMOSS/MOSS-TTS-Nano

使用官方 model.inference() API，兼容 voice_clone / continuation 两种模式。
"""

import numpy as np
from pathlib import Path
import tempfile
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
DEFAULT_AUDIO_TOKENIZER_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
DEFAULT_AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"


class MossTTSNanoAdapter(BaseTTSAdapter):
    model_type = "moss-tts-nano"
    display_name = "MOSS-TTS Nano (100M)"
    supported_languages = ["zh", "en", "ja", "ko", "fr", "de", "es"]
    default_sample_rate = 48000

    def __init__(self):
        super().__init__()
        self._audio_tokenizer_path = None

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
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "MOSS-TTS-Nano 需要安装:\n"
                "  pip install transformers>=4.45 torch>=2.0 torchaudio sentencepiece\n"
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

        # 加载主模型（使用 AutoModelForCausalLM，与官方 infer.py 一致）
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        self._model.eval()

        # 确定 audio tokenizer 路径
        if local_path is not None:
            codec_dir = local_path / "audio_tokenizer_nano"
            if (codec_dir / "config.json").exists():
                self._audio_tokenizer_path = str(codec_dir)
            else:
                self._audio_tokenizer_path = DEFAULT_AUDIO_TOKENIZER_ID
        else:
            self._audio_tokenizer_path = DEFAULT_AUDIO_TOKENIZER_ID

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        self._check_loaded()

        import torch

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

        text = request.text.strip()

        # 使用临时输出路径（inference() 会自动创建目录）
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self._model.inference(
                text=text,
                output_audio_path=tmp_path,
                mode="voice_clone",
                reference_audio_path=str(ref_path),
                audio_tokenizer_type=DEFAULT_AUDIO_TOKENIZER_TYPE,
                audio_tokenizer_pretrained_name_or_path=self._audio_tokenizer_path,
                device=self._device,
            )

            waveform = result["waveform"]
            sample_rate = result["sample_rate"]

            if isinstance(waveform, torch.Tensor):
                audio = waveform.detach().cpu().numpy()
            else:
                audio = np.asarray(waveform)

            return TTSResponse.from_numpy(audio, sample_rate)

        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": False,
            "emotion": False,
            "reference_audio": True,
        }
