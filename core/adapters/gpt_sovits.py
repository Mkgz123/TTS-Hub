"""GPT-SoVITS 适配器"""

import os
import sys
import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


class GPTSoVITSAdapter(BaseTTSAdapter):
    model_type = "gpt-sovits"
    display_name = "GPT-SoVITS"
    supported_languages = ["zh", "en", "ja"]
    default_sample_rate = 32000

    def __init__(self):
        super().__init__()
        self._api = None
        self._tts_fn = None

    def _import_gpt_sovits(self, model_path: str):
        """尝试导入 GPT-SoVITS"""
        import_errors = []
        path = Path(model_path)

        # 方法 1: 如果模型目录自带 GPT_SoVITS 包（如 RVC-Boss/GPT-SoVITS 克隆）
        gpt_sovits_dir = path / "GPT_SoVITS"
        if gpt_sovits_dir.exists():
            parent = str(path)
            if parent not in sys.path:
                sys.path.insert(0, parent)

        # 方法 2: 检查常见安装位置
        for candidate in [
            Path.home() / "GPT-SoVITS",
            Path("/opt/GPT-SoVITS"),
            Path("./GPT-SoVITS"),
        ]:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        # 尝试导入推理模块
        try:
            from GPT_SoVITS.inference_webui import get_tts_wav
            self._tts_fn = get_tts_wav
            self._import_mode = "webui"
            return
        except ImportError as e:
            import_errors.append(f"GPT_SoVITS.inference_webui: {e}")

        try:
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS as GPTSoVITSTTS
            self._tts_class = GPTSoVITSTTS
            self._import_mode = "class"
            return
        except ImportError as e:
            import_errors.append(f"GPT_SoVITS.TTS_infer_pack.TTS: {e}")

        try:
            from tools.i18n.i18n import I18n
            from infer import get_tts_wav as infer_tts
            self._tts_fn = infer_tts
            self._import_mode = "infer"
            return
        except ImportError as e:
            import_errors.append(f"infer module: {e}")

        error_msg = "\n".join(import_errors)
        raise ImportError(
            f"无法导入 GPT-SoVITS。请确保已正确安装：\n"
            f"1. 克隆仓库: git clone https://github.com/RVC-Boss/GPT-SoVITS\n"
            f"2. 安装依赖: cd GPT-SoVITS && pip install -r requirements.txt\n"
            f"3. 下载预训练模型到 GPT_SoVITS/pretrained_models/\n\n"
            f"导入错误详情:\n{error_msg}"
        )

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        """加载 GPT-SoVITS 模型

        model_path 可以是：
        - GPT-SoVITS 仓库根目录（包含 GPT_SoVITS/）
        - 包含 .ckpt 和 .pth 权重文件的目录
        """
        self._device = device
        self._model_path = Path(model_path)

        if not self._model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        self._import_gpt_sovits(model_path)

        # 查找权重文件
        gpt_files = list(self._model_path.glob("*.ckpt")) + \
                    list(self._model_path.glob("GPT_weights/*.ckpt")) + \
                    list(self._model_path.rglob("*.ckpt"))
        sovits_files = list(self._model_path.glob("*.pth")) + \
                       list(self._model_path.glob("SoVITS_weights/*.pth")) + \
                       list(self._model_path.rglob("*.pth"))

        # 过滤掉 pretrained_models 中的通用预训练权重
        gpt_files = [f for f in gpt_files if "pretrained" not in str(f)] or gpt_files
        sovits_files = [f for f in sovits_files if "pretrained" not in str(f)] or sovits_files

        self._gpt_path = str(gpt_files[0]) if gpt_files else ""
        self._sovits_path = str(sovits_files[0]) if sovits_files else ""

        if not self._gpt_path and not self._sovits_path:
            # 没有找到具体权重文件，但尝试用默认路径加载
            pretrained_dir = self._model_path / "GPT_SoVITS" / "pretrained_models"
            if pretrained_dir.exists():
                self._gpt_path = str(pretrained_dir)
                self._sovits_path = str(pretrained_dir)

        if self._import_mode == "class":
            try:
                from GPT_SoVITS.TTS_infer_pack.TTS import TTS as GPTSoVITSTTS
                self._model = GPTSoVITSTTS(device=device)
                if self._gpt_path:
                    self._model.init_t2s_weights(self._gpt_path)
                if self._sovits_path:
                    self._model.init_vits_weights(self._sovits_path)
            except Exception as e:
                raise RuntimeError(f"加载 GPT-SoVITS 模型失败: {e}")
        else:
            # webui/infer 模式下，tts_fn 会在调用时加载权重
            self._model = True  # 标记为已加载

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        self._check_loaded()

        # 参考音频：通过 speaker 或 extra 传递
        ref_audio = request.speaker or request.extra.get("ref_audio", "")
        ref_text = request.extra.get("ref_text", "大家好，我是AI助手。")
        prompt_text = request.extra.get("prompt_text", "")

        if not ref_audio:
            raise ValueError(
                "GPT-SoVITS 需要参考音频。\n"
                "请通过 speaker 参数或 extra['ref_audio'] 传入参考音频路径。"
            )

        if self._import_mode == "class":
            # 使用 TTS 类接口
            sr, audio = self._model.infer(
                ref_wav_path=ref_audio,
                prompt_text=ref_text or prompt_text,
                text=request.text,
                text_language=request.language,
            )
        else:
            # 使用 webui/infer 函数接口
            sr, audio = self._tts_fn(
                ref_wav_path=ref_audio,
                prompt_text=ref_text or prompt_text,
                text=request.text,
                text_language=request.language,
                gpt_path=self._gpt_path or None,
                sovits_path=self._sovits_path or None,
            )

        return TTSResponse.from_numpy(np.array(audio, dtype=np.float32), sr)

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": False,
            "emotion": False,
            "reference_audio": True,
        }
