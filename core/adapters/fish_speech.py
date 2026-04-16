"""Fish-Speech 适配器"""

import numpy as np
from pathlib import Path
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


class FishSpeechAdapter(BaseTTSAdapter):
    model_type = "fish-speech"
    display_name = "Fish-Speech"
    supported_languages = ["zh", "en"]
    default_sample_rate = 44100

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        self._device = device
        self._model_path = Path(model_path)
        
        # 尝试多种导入路径
        try:
            # 优先使用官方 fish_speech 包
            from fish_speech.inference_engine import FishSpeechInferenceEngine
            self._model = FishSpeechInferenceEngine(
                model_path=str(self._model_path),
                device=device,
            )
            self._engine_type = "fish_speech"
        except ImportError:
            try:
                # 备选: 使用 transformers + Qwen2ForCausalLM
                import torch
                from transformers import Qwen2ForCausalLM, AutoTokenizer
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(self._model_path), 
                    trust_remote_code=True
                )
                self._model = Qwen2ForCausalLM.from_pretrained(
                    str(self._model_path),
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device,
                )
                self._engine_type = "transformers"
            except ImportError:
                self._needs_package = True

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        if hasattr(self, "_needs_package"):
            raise NotImplementedError(
                "Fish-Speech 需要安装 fish_speech 包: pip install fish-speech\n"
                "或克隆仓库: git clone https://github.com/fishaudio/fish-speech\n"
                "备选方案: pip install transformers torch"
            )
        
        self._check_loaded()

        if self._engine_type == "fish_speech":
            # 使用官方 fish_speech 推理引擎
            audio = self._model.inference(
                text=request.text,
                speaker=request.speaker,
                language=request.language,
                speed=request.speed,
            )
        else:
            # transformers 模式下的基础推理（效果有限）
            # 这里提供一个基本的实现框架
            import torch
            
            inputs = self._tokenizer(request.text, return_tensors="pt")
            if self._device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 注意: 这只是基础推理，完整效果需要 fish_speech 官方库
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                )
            
            # 转换为音频（需要 fish_speech 的解码器）
            # 这里返回静音作为占位符
            import warnings
            warnings.warn(
                "transformers 模式下音频生成需要 fish_speech 官方解码器，返回静音",
                UserWarning
            )
            audio = np.zeros(int(self.default_sample_rate * 2.0), dtype=np.float32)

        return TTSResponse.from_numpy(
            np.array(audio, dtype=np.float32),
            self.default_sample_rate,
        )

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": True,
            "emotion": False,
            "reference_audio": True,
        }

    def get_speakers(self) -> list:
        """返回可用说话人列表"""
        if self.is_loaded and hasattr(self, "_engine_type"):
            if self._engine_type == "fish_speech" and hasattr(self._model, "list_speakers"):
                return list(self._model.list_speakers())
        return []