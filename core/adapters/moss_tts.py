"""MOSS-TTS 适配器"""

import numpy as np
from pathlib import Path
from typing import Optional, Any
from core.adapter_base import BaseTTSAdapter, TTSRequest, TTSResponse


class MossttsAdapter(BaseTTSAdapter):
    model_type = "moss-tts"
    display_name = "MOSS-TTS"
    supported_languages = ["zh"]
    default_sample_rate = 22050

    def __init__(self):
        super().__init__()
        self._moss_tts_class = None
        self._config = None
        self._custom_load = False

    def _import_moss_tts(self):
        """尝试多个 import 路径导入 MOSS-TTS"""
        import_errors = []
        
        # 尝试从 moss_tts 包导入
        try:
            from moss_tts import MOSSTTS
            self._moss_tts_class = MOSSTTS
            return True
        except ImportError as e:
            import_errors.append(f"moss_tts: {e}")
        
        # 尝试从 openmoss_tts 导入
        try:
            from openmoss_tts import MOSSTTS
            self._moss_tts_class = MOSSTTS
            return True
        except ImportError as e:
            import_errors.append(f"openmoss_tts: {e}")
        
        # 尝试从本地路径导入
        try:
            import sys
            possible_paths = [
                Path.home() / "MOSS-TTS",
                Path.home() / "moss-tts",
                Path("/opt/MOSS-TTS"),
                Path("./MOSS-TTS"),
            ]
            for p in possible_paths:
                if p.exists() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            
            from moss_tts import MOSSTTS
            self._moss_tts_class = MOSSTTS
            return True
        except ImportError as e:
            import_errors.append(f"local moss_tts: {e}")
        
        return False

    def _load_config(self, config_path: Optional[str] = None):
        """加载配置文件"""
        if config_path and Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                self._config = None

    def _torch_load_model(self, model_path: str, device: str = "cuda"):
        """使用 torch 直接加载模型权重"""
        import torch
        path = Path(model_path)
        
        # 查找所有可能的权重文件
        weight_extensions = ["*.pt", "*.pth", "*.bin", "*.safetensors"]
        weight_files = []
        for ext in weight_extensions:
            weight_files.extend(list(path.glob(ext)))
        
        if not weight_files:
            raise FileNotFoundError(f"未找到模型权重文件: {model_path}")
        
        # 加载权重
        self._weights = {}
        for weight_file in weight_files[:5]:  # 最多加载 5 个权重文件
            try:
                if weight_file.suffix == ".safetensors":
                    # 尝试加载 safetensors 格式
                    try:
                        from safetensors.torch import load_file
                        load_device = device if device != "cuda" else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
                        weights = load_file(str(weight_file), device=load_device)
                    except ImportError:
                        print("需要安装 safetensors: pip install safetensors")
                        continue
                else:
                    # PyTorch 格式 — map_location 自动处理 device
                    weights = torch.load(
                        str(weight_file),
                        map_location=device,
                        weights_only=True
                    )
                
                if isinstance(weights, dict):
                    self._weights.update(weights)
                else:
                    self._weights[weight_file.stem] = weights
                    
            except Exception as e:
                print(f"加载权重文件 {weight_file} 失败: {e}")
        
        if not self._weights:
            raise RuntimeError("未能加载任何模型权重")
        
        self._custom_load = True
        self._device = device

    def load_model(self, model_path: str, device: str = "cuda") -> None:
        """加载 MOSS-TTS 模型

        Args:
            model_path: 模型目录路径
            device: 运行设备 ("cuda" 或 "cpu")
        """
        # 自动检测 CUDA 可用性
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[MOSS-TTS] CUDA 不可用，回退到 CPU")
                    device = "cpu"
            except ImportError:
                device = "cpu"

        self._device = device
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 查找配置文件
        config_files = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
        config_path = str(config_files[0]) if config_files else None
        self._load_config(config_path)
        
        # 尝试导入 MOSS-TTS 包
        if self._import_moss_tts():
            try:
                self._model = self._moss_tts_class(
                    config=config_path,
                    model_path=model_path,
                    device=device
                )
                return
            except Exception as e:
                print(f"使用 MOSS-TTS 包加载失败，尝试直接加载权重: {e}")
        
        # 回退：使用 torch 直接加载权重
        try:
            self._torch_load_model(model_path, device)
        except Exception as e:
            raise RuntimeError(
                f"加载 MOSS-TTS 模型失败。\n"
                f"请确保已安装 MOSS-TTS:\n"
                f"1. 克隆仓库: git clone https://github.com/OpenMOSS/MOSS-TTS\n"
                f"2. 安装依赖: cd MOSS-TTS && pip install -r requirements.txt\n"
                f"3. 或者确保模型目录包含有效的权重文件\n\n"
                f"错误详情: {e}"
            )

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """合成语音
        
        MOSS-TTS 支持中文语音合成
        """
        # 检查模型是否加载
        if not self._custom_load:
            self._check_loaded()
        
        # 如果使用自定义加载（直接加载权重）
        if self._custom_load:
            raise NotImplementedError(
                "MOSS-TTS 需要完整的推理代码支持。\n"
                "请安装 MOSS-TTS 包:\n"
                "1. 克隆仓库: git clone https://github.com/OpenMOSS/MOSS-TTS\n"
                "2. 安装依赖: cd MOSS-TTS && pip install -r requirements.txt\n"
                "3. 确保模型目录包含必要的配置和权重文件"
            )
        
        # 准备合成参数
        synthesis_kwargs = {
            "text": request.text,
        }
        
        # 添加语速参数（如果支持）
        if hasattr(self._model, 'synthesize'):
            # 标准接口
            try:
                audio = self._model.synthesize(
                    text=request.text,
                    speed=request.speed,
                    **request.extra
                )
            except TypeError:
                # 尝试不带 speed 参数
                audio = self._model.synthesize(text=request.text)
        elif hasattr(self._model, 'generate'):
            # 另一种可能的接口
            audio = self._model.generate(text=request.text)
        elif hasattr(self._model, '__call__'):
            # 直接调用
            audio = self._model(text=request.text)
        else:
            raise RuntimeError("模型不支持任何已知的合成接口")
        
        # 转换为 numpy 数组，确保是 float32
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
        elif hasattr(audio, 'numpy'):
            audio = audio.numpy().astype(np.float32)
        elif hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy().astype(np.float32)
        elif isinstance(audio, (list, tuple)):
            audio = np.array(audio, dtype=np.float32)
        else:
            # 尝试转换为 numpy 数组
            try:
                audio = np.array(audio, dtype=np.float32)
            except:
                raise RuntimeError(f"无法将音频数据转换为 numpy 数组: {type(audio)}")
        
        # 确保音频是一维数组
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # 应用语速调整（如果需要且模型不支持）
        speed = request.speed
        if speed != 1.0:
            # 简单的重采样方法调整语速
            target_length = int(len(audio) / speed)
            indices = np.linspace(0, len(audio) - 1, target_length)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        
        return TTSResponse.from_numpy(audio, self.default_sample_rate)

    def get_supported_features(self) -> dict:
        return {
            "speaker": False,  # MOSS-TTS 通常不支持说话人选择
            "language": False,  # 主要支持中文
            "speed": True,      # 支持语速调节
            "emotion": False,   # 不支持情感控制
            "reference_audio": False,  # 不支持参考音频
        }

    def get_speakers(self) -> list:
        """获取可用的说话人列表
        
        MOSS-TTS 通常不提供预设说话人列表
        """
        return []