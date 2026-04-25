"""MOSS-TTS 适配器 (v1.0)

基于 HuggingFace transformers 的 MOSS-TTSD v1.0 模型适配。
对话式语音合成，支持零样本声音克隆，需要提供参考音频。

模型: OpenMOSS-Team/MOSS-TTSD-v1.0
仓库: https://github.com/OpenMOSS/MOSS-TTS
"""

import importlib.util
import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-TTSD-v1.0"


class MossttsAdapter(BaseTTSAdapter):
    model_type = "moss-tts"
    display_name = "MOSS-TTS v1.0"
    supported_languages = ["zh", "en"]
    default_sample_rate = 24000

    def __init__(self):
        super().__init__()
        self._processor = None

    def _resolve_attn_implementation(self, device: str, dtype) -> str:
        """选择合适的 attention 实现"""
        import torch
        if (
            device == "cuda"
            and importlib.util.find_spec("flash_attn") is not None
            and dtype in {torch.float16, torch.bfloat16}
        ):
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return "flash_attention_2"
        if device == "cuda":
            return "sdpa"
        return "eager"

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
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise ImportError(
                "MOSS-TTS (TTSD v1.0) 需要安装:\n"
                "  pip install transformers>=4.45 torch>=2.0 torchaudio\n"
                "或在 WebUI 环境管理中选择 moss-tts 重新创建环境"
            )

        # CUDA SDPA 后端配置
        if device == "cuda":
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        # 确定模型路径
        local_path = None
        if path.exists() and (path / "config.json").exists():
            model_id = str(path)
            local_path = path
        else:
            model_id = model_path if "/" in model_path else DEFAULT_MODEL_ID

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        attn_implementation = self._resolve_attn_implementation(device, dtype)

        # 如果本地有 audio_tokenizer 子目录，优先使用本地文件
        processor_kwargs = {"trust_remote_code": True}
        if local_path is not None:
            codec_dir = local_path / "audio_tokenizer"
            if (codec_dir / "config.json").exists():
                processor_kwargs["codec_path"] = str(codec_dir)

        self._processor = AutoProcessor.from_pretrained(
            model_id,
            **processor_kwargs,
        )

        # 检测模型类型：TTSD v1.0 有 audio_tokenizer，Nano 系列没有
        if not hasattr(self._processor, "audio_tokenizer"):
            raise RuntimeError(
                "检测到 MOSS-TTS Nano 系列模型，当前适配器仅支持 TTSD v1.0。\n\n"
                "请下载正确的模型：\n"
                "  HuggingFace: OpenMOSS-Team/MOSS-TTSD-v1.0\n"
                "或在 WebUI 下载选项卡中选择「MOSS-TTSD v1.0 对话合成 (推荐)」\n\n"
                f"当前加载的模型: {model_id}"
            )

        if device == "cuda":
            self._processor.audio_tokenizer = self._processor.audio_tokenizer.to(device)
        self._processor.audio_tokenizer.eval()

        self._model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
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
                "MOSS-TTS 需要参考音频进行声音克隆。\n"
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
