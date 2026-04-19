"""F5-TTS 适配器"""

import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


class F5TTSAdapter(BaseTTSAdapter):
    model_type = "f5-tts"
    display_name = "F5-TTS"
    supported_languages = ["zh", "en"]
    default_sample_rate = 24000

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        self._device = device
        self._model_path = Path(model_path)
        
        try:
            from f5_tts.api import F5TTS
        except ImportError:
            self._needs_package = True
            return
        
        # 查找 checkpoint 文件
        ckpt_file = ""
        for pattern in ["*.safetensors", "*.pt", "*.pth", "*.ckpt"]:
            found = list(self._model_path.glob(pattern))
            if found:
                ckpt_file = str(found[0])
                break
        
        # 查找 vocab 文件
        vocab_file = ""
        vocab_found = list(self._model_path.glob("*.txt")) + list(self._model_path.glob("*.json"))
        for vf in vocab_found:
            if "vocab" in vf.name.lower():
                vocab_file = str(vf)
                break
        
        # 查找模型配置文件 (YAML)
        model_name = "F5TTS_v1_Base"  # 默认
        yaml_found = list(self._model_path.glob("*.yaml")) + list(self._model_path.glob("*.yml"))
        if yaml_found:
            # 用找到的 yaml 文件名（不含扩展名）作为 model 配置名
            model_name = yaml_found[0].stem
        
        try:
            self._model = F5TTS(
                model=model_name,
                ckpt_file=ckpt_file,
                vocab_file=vocab_file,
                device=device,
            )
        except Exception as e:
            raise RuntimeError(f"加载 F5-TTS 模型失败: {e}")

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        if hasattr(self, "_needs_package"):
            raise NotImplementedError(
                "F5-TTS 需要安装: pip install f5-tts\n"
                "或克隆仓库: git clone https://github.com/SWivid/F5-TTS"
            )
        
        self._check_loaded()

        # 获取参考音频和文本
        ref_audio = request.extra.get("ref_audio", "")
        ref_text = request.extra.get("ref_text", "")
        
        # 如果提供了 speaker 且看起来像文件路径，用作参考音频
        # 去掉可能的外层引号（repr() 嵌入 f-string 时会产生）
        ref_path = request.speaker
        if ref_path and len(ref_path) >= 2 and ref_path[0] == "'" and ref_path[-1] == "'":
            ref_path = ref_path[1:-1]
        if ref_path and Path(ref_path).exists():
            ref_audio = ref_path

        # 调用 F5-TTS 推理
        # 注意: 返回值可能是 (audio, sr) 或 (audio, sr, spectogram)
        result = self._model.infer(
            ref_file=ref_audio if ref_audio else None,
            ref_text=ref_text if ref_text else "",
            gen_text=request.text,
            speed=request.speed,
        )
        
        # 处理不同版本的返回值
        if len(result) == 2:
            audio, sr = result
        elif len(result) == 3:
            audio, sr, _ = result
        else:
            raise RuntimeError(f"Unexpected return value from F5-TTS infer: {len(result)} values")

        return TTSResponse.from_numpy(np.array(audio, dtype=np.float32), sr)

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": True,
            "emotion": False,
            "reference_audio": True,
        }