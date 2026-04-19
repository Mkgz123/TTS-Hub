"""CosyVoice 适配器"""

import numpy as np
from pathlib import Path
from typing import Optional, Generator
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


class CosyVoiceAdapter(BaseTTSAdapter):
    model_type = "cosyvoice"
    display_name = "CosyVoice"
    supported_languages = ["zh", "en"]
    default_sample_rate = 22050

    def __init__(self):
        super().__init__()
        self._cosyvoice_class = None

    def _import_cosyvoice(self):
        """尝试多个 import 路径导入 CosyVoice"""
        import_errors = []
        
        # 尝试从 cosyvoice 包导入
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
            self._cosyvoice_class = CosyVoice
            return
        except ImportError as e:
            import_errors.append(f"cosyvoice.cli.cosyvoice: {e}")
        
        # 尝试从 cosyvoice_cli 导入
        try:
            from cosyvoice_cli import CosyVoice
            self._cosyvoice_class = CosyVoice
            return
        except ImportError as e:
            import_errors.append(f"cosyvoice_cli: {e}")
        
        # 尝试从当前目录导入（如果用户克隆了仓库到本地）
        try:
            import sys
            from pathlib import Path
            # 添加可能的本地路径
            possible_paths = [
                Path.home() / "CosyVoice",
                Path("/opt/CosyVoice"),
                Path("./CosyVoice"),
                # 添加 envs 目录下的路径（post_install 克隆的位置）
                Path(__file__).parent.parent.parent / "envs" / "cosyvoice" / "CosyVoice",
            ]
            for p in possible_paths:
                if p.exists() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            
            from cosyvoice.cli.cosyvoice import CosyVoice
            self._cosyvoice_class = CosyVoice
            return
        except ImportError as e:
            import_errors.append(f"local cosyvoice: {e}")
        
        # 所有导入都失败
        error_msg = "\n".join(import_errors)
        raise ImportError(
            f"无法导入 CosyVoice。请确保已正确安装：\n"
            f"1. 克隆仓库: git clone https://github.com/FunAudioLLM/CosyVoice\n"
            f"2. 安装依赖: cd CosyVoice && pip install -r requirements.txt\n"
            f"3. 或者确保 cosyvoice 包在 Python 路径中\n\n"
            f"导入错误详情:\n{error_msg}"
        )

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        """加载 CosyVoice 模型
        
        Args:
            model_path: 模型目录路径，应包含 cosyvoice.yaml, llm.pt, speech_tokenizer.pt
            device: 运行设备 ("cuda" 或 "cpu")
        """
        self._device = device
        self._import_cosyvoice()
        
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 检查必要的模型文件
        required_files = ["cosyvoice.yaml", "llm.pt", "speech_tokenizer.pt"]
        missing_files = []
        for f in required_files:
            if not (path / f).exists():
                missing_files.append(f)
        
        if missing_files:
            raise FileNotFoundError(
                f"模型目录缺少必要文件: {', '.join(missing_files)}\n"
                f"请确保模型目录包含: cosyvoice.yaml, llm.pt, speech_tokenizer.pt"
            )
        
        try:
            self._model = self._cosyvoice_class(str(path), device=device)
        except Exception as e:
            raise RuntimeError(f"加载 CosyVoice 模型失败: {e}")

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """合成语音
        
        支持的模式：
        1. 零样本声音克隆: request.extra["mode"] == "zero_shot" 且提供 prompt_audio
        2. 指定说话人: request.speaker 不为空
        3. 默认合成: 使用默认说话人
        """
        self._check_loaded()
        
        audio_tensor = None
        
        try:
            # 零样本声音克隆模式
            if (request.extra.get("mode") == "zero_shot" and 
                request.extra.get("prompt_audio")):
                
                prompt_audio = request.extra["prompt_audio"]
                prompt_text = request.extra.get("prompt_text", "")
                
                # 确保 prompt_audio 是文件路径或音频数据
                if isinstance(prompt_audio, str):
                    # 假设是文件路径
                    results = self._model.inference_zero_shot(
                        request.text,
                        prompt_audio,
                        prompt_text,
                    )
                else:
                    # 假设是音频数据，需要保存为临时文件
                    import tempfile
                    import soundfile as sf
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        sf.write(f.name, prompt_audio, self.default_sample_rate)
                        results = self._model.inference_zero_shot(
                            request.text,
                            f.name,
                            prompt_text,
                        )
                
                # 获取第一个结果的音频
                for result in results:
                    audio_tensor = result["tts_speech"]
                    break
            
            # 指定说话人模式
            elif request.speaker:
                results = self._model.inference_sft(request.text, request.speaker)
                for result in results:
                    audio_tensor = result["tts_speech"]
                    break
            
            # 默认模式
            else:
                # 尝试获取默认说话人
                default_speaker = ""
                if hasattr(self._model, "list_avaliable_spks"):
                    spks = list(self._model.list_avaliable_spks())
                    if spks:
                        default_speaker = spks[0]
                
                results = self._model.inference_sft(request.text, default_speaker)
                for result in results:
                    audio_tensor = result["tts_speech"]
                    break
            
            if audio_tensor is None:
                raise RuntimeError("合成失败：未获取到音频数据")
            
            # 转换为 numpy 数组，确保是 float32
            if hasattr(audio_tensor, 'numpy'):
                audio = audio_tensor.numpy().flatten()
            elif hasattr(audio_tensor, 'cpu'):
                audio = audio_tensor.cpu().numpy().flatten()
            else:
                audio = np.array(audio_tensor).flatten()
            
            audio = audio.astype(np.float32)
            
            return TTSResponse.from_numpy(audio, self.default_sample_rate)
            
        except Exception as e:
            raise RuntimeError(f"合成失败: {e}")

    def get_supported_features(self) -> dict:
        return {
            "speaker": True,
            "language": True,
            "speed": False,  # CosyVoice 标准 API 不直接支持语速调节
            "emotion": False,
            "reference_audio": True,  # 支持零样本声音克隆
        }

    def get_speakers(self) -> list:
        """获取可用的说话人列表"""
        self._check_loaded()
        if hasattr(self._model, "list_avaliable_spks"):
            return list(self._model.list_avaliable_spks())
        return []