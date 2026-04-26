"""TTS 音频后处理工具"""

import numpy as np


def detect_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = -40.0,
    min_speech_ms: int = 100,
    min_silence_ms: int = 200,
) -> list[tuple[int, int]]:
    """检测语音段（非静音区域）

    Returns:
        list of (start_sample, end_sample) 语音段列表
    """
    if audio.ndim > 1:
        audio = audio.reshape(-1)

    frame_ms = 30
    frame_size = int(sample_rate * frame_ms / 1000)
    hop_size = frame_size // 2

    if len(audio) < frame_size:
        return [(0, len(audio))]

    num_frames = (len(audio) - frame_size) // hop_size + 1

    # 每帧 RMS 能量
    rms = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = audio[start:end]
        rms[i] = np.sqrt(np.mean(frame ** 2))

    # 能量阈值
    ref = max(np.max(np.abs(audio)), 1e-10)
    threshold = ref * (10 ** (silence_threshold_db / 20))

    is_speech = rms > threshold

    # 合并临近语音段
    min_speech_frames = max(1, int(min_speech_ms / frame_ms))
    min_silence_frames = max(1, int(min_silence_ms / frame_ms))

    # 形态学：去除过短的语音/静音
    segments = []
    in_speech = False
    speech_start = 0
    silence_count = 0

    for i in range(num_frames):
        if is_speech[i]:
            silence_count = 0
            if not in_speech:
                speech_start = i * hop_size
                in_speech = True
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    # 结束当前语音段
                    speech_end = (i - silence_count + 1) * hop_size + frame_size
                    speech_end = min(speech_end, len(audio))

                    # 检查语音段是否足够长
                    if speech_end - speech_start >= min_speech_ms * sample_rate / 1000:
                        segments.append((speech_start, speech_end))
                    in_speech = False
                    silence_count = 0

    if in_speech:
        speech_end = len(audio)
        if speech_end - speech_start >= min_speech_ms * sample_rate / 1000:
            segments.append((speech_start, speech_end))

    return segments


def adjust_sentence_gap(
    audio: np.ndarray,
    sample_rate: int,
    gap_ms: float = 0.0,
    silence_threshold_db: float = -40.0,
) -> np.ndarray:
    """调整句子之间的静音间隙

    检测语音段，统一处理段间静音。

    Args:
        audio: 原始音频 (1D numpy array)
        sample_rate: 采样率
        gap_ms:
            > 0  → 所有句间间隙设为精确值（毫秒），统一替换原始静音
            = 0  → 不做任何调整，返回原始音频
            < 0  → 每个原始间隙缩减 abs(gap_ms) 毫秒（不下溢到 0 以下）
        silence_threshold_db: 静音阈值（dB）

    Returns:
        调整后的音频
    """
    if gap_ms == 0:
        return audio

    if audio.ndim > 1:
        audio = audio.reshape(-1)

    segments = detect_speech_segments(audio, sample_rate, silence_threshold_db)

    if len(segments) <= 1:
        return audio  # 只有一段或没有语音，无需调整

    if gap_ms > 0:
        # === 正数模式：统一替换为固定间隙 ===
        gap_samples = int(sample_rate * gap_ms / 1000)
        total_len = sum(end - start for start, end in segments) + gap_samples * (len(segments) - 1)
        result = np.zeros(total_len, dtype=audio.dtype)
        pos = 0
        for i, (start, end) in enumerate(segments):
            seg_len = end - start
            result[pos:pos + seg_len] = audio[start:end]
            pos += seg_len
            if i < len(segments) - 1:
                pos += gap_samples
        return result

    else:
        # === 负数模式：从原始间隙中缩减 ===
        shrink_samples = int(sample_rate * abs(gap_ms) / 1000)

        # 计算每个原始间隙并缩减
        new_gaps = []
        for i in range(len(segments) - 1):
            orig_gap = segments[i + 1][0] - segments[i][1]
            new_gap = max(0, orig_gap - shrink_samples)
            new_gaps.append(new_gap)

        total_len = sum(end - start for start, end in segments) + sum(new_gaps)
        result = np.zeros(total_len, dtype=audio.dtype)

        pos = 0
        for i, (start, end) in enumerate(segments):
            seg_len = end - start
            result[pos:pos + seg_len] = audio[start:end]
            pos += seg_len
            if i < len(segments) - 1:
                pos += new_gaps[i]

        return result
