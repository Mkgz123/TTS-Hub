"""MOSS-TTS 适配器"""

import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


class MossttsAdapter(BaseTTSAdapter):
    model_type = "moss-tts"
    display_name = "MOSS-TTS"
    supported_languages = ["zh"]
    default_sample_rate = 22050

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        self._device = device
        path = Path(model_path)
        self._model_path = path

        # MOSS-TTS 通常以 yaml config + checkpoint 形式分发
        yaml_files = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
        pt_files = list(path.glob("*.pt")) + list(path.glob("*.pth")) + list(path.glob("*.bin"))

        if not pt_files:
            raise FileNotFoundError(f"未找到模型权重文件: {model_path}")

        try:
            # 尝试从 OpenMOSS 包加载
            from moss_tts import MOSSTTS
            config = str(yaml_files[0]) if yaml_files else None
            self._model = MOSSTTS(config=config, model_path=model_path, device=device)
        except ImportError:
            # Fallback: 用 torch 直接加载
            import torch
            self._weights = {}
            for pt_file in pt_files[:3]:  # 最多加载 3 个权重文件
                self._weights[pt_file.stem] = torch.load(
                    str(pt_file), map_location=device, weights_only=True
                )
            self._custom_load = True

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        if not self.is_loaded and not hasattr(self, "_custom_load"):
            raise RuntimeError("模型未加载")

        if hasattr(self, "_custom_load"):
            raise NotImplementedError(
                "MOSS-TTS 需要安装 moss_tts 包或克隆仓库:\n"
                "git clone https://github.com/OpenMOSS/MOSS-TTS\n"
                "参考 README 安装依赖"
            )

        audio = self._model.synthesize(
            text=request.text,
            speed=request.speed,
        )
        return TTSResponse.from_numpy(
            np.array(audio, dtype=np.float32),
            self.default_sample_rate,
        )

    def get_supported_features(self) -> dict:
        return {
            "speaker": False,
            "language": False,
            "speed": True,
            "emotion": False,
            "reference_audio": False,
        }
