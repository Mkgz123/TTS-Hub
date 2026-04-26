"""MOSS-TTS-Realtime 适配器

流式 TTS 模型，支持零样本声音克隆 + 多轮流式合成，32K 上下文。
架构名 MossTTSRealtime（不同于 moss_tts_delay），使用自定义模型类 + AutoTokenizer + 独立 codec。

模型: OpenMOSS-Team/MOSS-TTS-Realtime (1.7B)
仓库: https://github.com/OpenMOSS/MOSS-TTS

注意: 仅支持 CUDA。
"""

import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Realtime"
DEFAULT_CODEC_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer"


class MossTTSRealtimeAdapter(BaseTTSAdapter):
    model_type = "moss-tts-realtime"
    display_name = "MOSS-TTS-Realtime"
    supported_languages = ["zh", "en", "ja", "ko", "de", "fr", "es", "pt", "it", "ru"]
    default_sample_rate = 24000

    def __init__(self):
        super().__init__()
        self._tokenizer = None
        self._codec = None
        self._inferencer = None

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        import sys

        # MOSS-TTS 仓库路径 — post_install 克隆到 envs/<model>/MOSS-TTS
        _project_root = Path(__file__).parent.parent.parent
        _moss_dir = _project_root / "envs" / "moss-tts-realtime" / "MOSS-TTS" / "moss_tts_realtime"
        if _moss_dir.exists() and str(_moss_dir) not in sys.path:
            sys.path.insert(0, str(_moss_dir))

        try:
            import torch
            import torchaudio
            # Monkey-patch torchaudio.load → soundfile，torchaudio 2.11+ 默认走 torchcodec 后端
            import soundfile as _sf
            _torchaudio_load = torchaudio.load
            def _patched_load(uri, *args, **kwargs):
                data, sr = _sf.read(uri, dtype="float32", always_2d=False)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                else:
                    data = data.T
                return torch.from_numpy(data.copy()), sr
            torchaudio.load = _patched_load

            from transformers import AutoTokenizer, AutoModel
            from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
            from inferencer import MossTTSRealtimeInference
        except ImportError:
            raise ImportError(
                "MOSS-TTS-Realtime 需要安装 moss_tts_realtime 包:\n\n"
                "方法 1 (推荐): 在 WebUI「环境管理」Tab 中:\n"
                "  ① 选择 moss-tts-realtime → 点击「删除环境」\n"
                "  ② 再点击「创建环境」→ 等待 post_install 完成\n\n"
                "方法 2 (手动): 激活 conda 环境后运行:\n"
                "  git clone https://github.com/OpenMOSS/MOSS-TTS.git\n"
                "  pip install git+https://github.com/OpenMOSS/MOSS-TTS.git"
            )

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "MOSS-TTS-Realtime 需要 CUDA GPU。\n"
                "当前系统未检测到 NVIDIA GPU，无法运行此模型。"
            )
        self._device = device

        path = Path(model_path)
        if path.exists() and (path / "config.json").exists():
            model_id = str(path)
        else:
            model_id = model_path if "/" in model_path else DEFAULT_MODEL_ID

        # 查找本地 codec 路径
        codec_subdir = path / "audio_tokenizer"
        if codec_subdir.exists() and (codec_subdir / "config.json").exists():
            codec_path = str(codec_subdir)
        else:
            # 也尝试检查 models/ 下的独立 audio_tokenizer 目录
            alt_codec = path.parent / "audio_tokenizer"
            if alt_codec.exists() and (alt_codec / "config.json").exists():
                codec_path = str(alt_codec)
            else:
                codec_path = DEFAULT_CODEC_ID

        dtype = torch.bfloat16

        # CUDA SDPA 后端
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        # 1. Tokenizer（不是 AutoProcessor）
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        # 2. 自定义模型类
        self._model = MossTTSRealtime.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            torch_dtype=dtype,
        ).to(device)
        self._model.eval()

        # 3. 独立 Audio Codec
        self._codec = AutoModel.from_pretrained(
            codec_path,
            trust_remote_code=True,
        ).eval().to(device)

        # 4. Inferencer
        self._inferencer = MossTTSRealtimeInference(
            model=self._model,
            tokenizer=self._tokenizer,
            max_length=5000,
            codec=self._codec,
            codec_sample_rate=24000,
            codec_encode_kwargs={"chunk_duration": 8},
        )

        self._sample_rate = 24000

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        self._check_loaded()

        import torch

        text = (request.text or "").strip()
        if not text:
            raise ValueError("请输入合成文本。\n每行一段对话，例如:\n你好，今天天气不错。\n是啊，我们去公园散步吧。")

        # 按换行分割为多段
        texts = [t.strip() for t in text.split("\n") if t.strip()]

        # 参考音频 — 优先使用 ref_audio_list（每段对应一个参考音频）
        ref_audio_list = request.extra.get("ref_audio_list") or None
        ref_audio = request.extra.get("ref_audio") or None
        if ref_audio_list:
            # 确保长度匹配
            reference_audio_path = ref_audio_list[:len(texts)]
            if len(reference_audio_path) < len(texts):
                reference_audio_path.extend([None] * (len(texts) - len(reference_audio_path)))
            # 过滤 None 为空字符串
            reference_audio_path = [r if r else None for r in reference_audio_path]
        elif ref_audio:
            reference_audio_path = [ref_audio] * len(texts)
        else:
            reference_audio_path = None

        # 采样参数
        temperature = float(request.extra.get("temperature", 0.8))
        top_p = float(request.extra.get("top_p", 0.6))
        top_k = int(request.extra.get("top_k", 30))
        repetition_penalty = float(request.extra.get("repetition_penalty", 1.1))
        repetition_window = int(request.extra.get("repetition_window", 50))

        with torch.no_grad():
            # generate() 返回 List[List[int]] — 原始 audio tokens
            token_sequences = self._inferencer.generate(
                text=texts,
                reference_audio_path=reference_audio_path,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                device=self._device,
            )

        # 解码每段 tokens → 音频
        audio_parts = []
        for tokens in token_sequences:
            tokens_tensor = torch.tensor(tokens, device=self._device)
            if tokens_tensor.ndim == 1:
                tokens_tensor = tokens_tensor.unsqueeze(0)
            # codec.decode 期望 (seq_len, num_codebooks)
            decode_result = self._codec.decode(
                tokens_tensor.permute(1, 0),
                chunk_duration=8,
            )
            wav = decode_result["audio"][0].detach().cpu().to(torch.float32)
            audio_parts.append(wav)

        # 拼接所有段落
        if len(audio_parts) == 1:
            audio = audio_parts[0].numpy()
        else:
            audio = torch.cat(audio_parts, dim=-1).numpy()

        return TTSResponse.from_numpy(audio.astype(np.float32), self._sample_rate)

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,       # 参考音频（声音克隆）
            "language": True,
            "speed": False,
            "emotion": False,
            "reference_audio": True,
            "instruction": False,
            "ambient_sound": False,
        }
