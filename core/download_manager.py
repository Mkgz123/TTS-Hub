"""
模型下载管理器
通过 HuggingFace Hub 下载和管理 TTS 模型
"""

import os
import json
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field


# 知名模型的 HuggingFace repo 映射
KNOWN_MODELS = {
    "fish-speech": {
        "fish-speech/fish-speech-1.5": "Fish-Speech 1.5 (推荐)",
        "fish-speech/fish-speech-1.4": "Fish-Speech 1.4",
        "fish-speech/fish-speech-s1": "Fish-Speech S1",
    },
    "f5-tts": {
        "SWivid/F5-TTS": "F5-TTS Base",
        "MikhailRoshchin/F5TTS_v1": "F5-TTS v1",
    },
    "chattts": {
        "2noise/ChatTTS": "ChatTTS 官方",
    },
    "cosyvoice": {
        "FunAudioLLM/CosyVoice-300M": "CosyVoice 300M",
        "FunAudioLLM/CosyVoice-300M-SFT": "CosyVoice 300M SFT",
        "FunAudioLLM/CosyVoice-300M-Instruct": "CosyVoice 300M Instruct",
    },
    "kokoro": {
        "hexgrad/Kokoro-82M": "Kokoro 82M",
        "hexgrad/Kokoro-82M-v1.1-zh": "Kokoro v1.1 中文",
    },
    "xtts": {
        "coqui/XTTS-v2": "XTTSv2 官方",
    },
    "moss-tts": {
        "OpenMOSS-Team/MOSS-TTSD-v1.0": "MOSS-TTSD v1.0 对话合成 (推荐)",
    },
    "moss-tts-nano": {
        "OpenMOSS-Team/MOSS-TTS-Nano-100M": "MOSS-TTS Nano 100M (轻量 CPU)",
        "OpenMOSS-Team/MOSS-TTS-Nano": "MOSS-TTS Nano 多语言",
    },
    "gpt-sovits": {
        "RVC-Boss/GPT-SoVITS": "GPT-SoVITS 官方预训练",
    },
}

# 某些模型的推理代码引用了其他 HF 仓库，需要一并下载到本地
# 格式: model_type → {repo_id: 子目录名}
RELATED_REPOS = {
    "moss-tts": {
        "OpenMOSS-Team/MOSS-Audio-Tokenizer": "audio_tokenizer",
    },
    "moss-tts-nano": {
        "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano": "audio_tokenizer_nano",
    },
}


@dataclass
class DownloadProgress:
    """下载进度"""
    repo_id: str
    status: str = "pending"  # pending, downloading, completed, error
    progress: float = 0.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error: Optional[str] = None


class DownloadManager:
    """模型下载管理器"""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._active_downloads: dict[str, DownloadProgress] = {}

    def list_known_models(self, model_type: Optional[str] = None) -> dict:
        """列出已知模型"""
        if model_type:
            return {model_type: KNOWN_MODELS.get(model_type, {})}
        return KNOWN_MODELS

    def list_local_models(self) -> list[dict]:
        """列出本地已下载的模型"""
        if not self.model_dir.exists():
            return []

        results = []
        for item in sorted(self.model_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                # 读取 model_info.json（如果有）
                info_file = item / "model_info.json"
                info = {}
                if info_file.exists():
                    try:
                        info = json.loads(info_file.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

                results.append({
                    "name": item.name,
                    "path": str(item),
                    "size_mb": self._get_dir_size(item) / (1024 * 1024),
                    "info": info,
                })
        return results

    def download_model(
        self,
        repo_id: str,
        model_type: Optional[str] = None,
        revision: Optional[str] = None,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """从 HuggingFace 下载模型

        Args:
            repo_id: HuggingFace repo ID（如 "fish-speech/fish-speech-1.5"）
            model_type: 模型类型（用于命名目录）
            revision: 分支/标签
            include_patterns: 包含的文件模式
            exclude_patterns: 排除的文件模式
            progress_callback: 进度回调 fn(repo_id, progress_pct)

        Returns:
            dict: {"success": bool, "path": str, "error": str}
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return {
                "success": False,
                "error": "需要安装 huggingface_hub: pip install huggingface_hub",
            }

        # 确定本地目录名
        safe_name = repo_id.replace("/", "__")
        local_dir = self.model_dir / safe_name

        progress = DownloadProgress(repo_id=repo_id, status="downloading")
        self._active_downloads[repo_id] = progress

        try:
            kwargs = {
                "repo_id": repo_id,
                "local_dir": str(local_dir),
                "local_dir_use_symlinks": False,
            }
            if revision:
                kwargs["revision"] = revision
            if include_patterns:
                kwargs["allow_patterns"] = include_patterns
            if exclude_patterns:
                kwargs["ignore_patterns"] = exclude_patterns

            snapshot_download(**kwargs)

            # 下载关联仓库（如 audio tokenizer）
            related = RELATED_REPOS.get(model_type or "", {})
            for related_repo_id, subdir in related.items():
                try:
                    related_dir = local_dir / subdir
                    related_dir.mkdir(parents=True, exist_ok=True)
                    snapshot_download(
                        repo_id=related_repo_id,
                        local_dir=str(related_dir),
                        local_dir_use_symlinks=False,
                    )
                except Exception:
                    pass  # 关联仓库下载失败不阻塞主流程

            progress.status = "completed"
            progress.progress = 100.0

            # 保存模型信息
            info = {
                "repo_id": repo_id,
                "model_type": model_type or "unknown",
                "source": "huggingface",
            }
            info_file = local_dir / "model_info.json"
            info_file.write_text(
                json.dumps(info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            return {
                "success": True,
                "path": str(local_dir),
                "name": safe_name,
            }

        except Exception as e:
            progress.status = "error"
            progress.error = str(e)
            return {
                "success": False,
                "error": f"下载失败: {e}",
            }
        finally:
            self._active_downloads.pop(repo_id, None)

    def get_download_status(self, repo_id: Optional[str] = None) -> dict:
        """获取下载状态"""
        if repo_id:
            progress = self._active_downloads.get(repo_id)
            if progress:
                return {"repo_id": repo_id, "status": progress.status,
                        "progress": progress.progress, "error": progress.error}
            return {"repo_id": repo_id, "status": "not_found"}
        return {k: {"status": v.status, "progress": v.progress}
                for k, v in self._active_downloads.items()}

    def remove_model(self, model_name: str) -> dict:
        """删除本地模型"""
        import shutil
        target = self.model_dir / model_name
        if target.exists():
            shutil.rmtree(str(target))
            return {"success": True, "message": f"已删除: {model_name}"}
        return {"success": False, "error": f"模型不存在: {model_name}"}

    @staticmethod
    def _get_dir_size(path: Path) -> int:
        """计算目录大小（字节）"""
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
