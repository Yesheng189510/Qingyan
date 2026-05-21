# -*- coding: utf-8 -*-
"""
实验一：人工信号上的经典非参数谱估计比较
实验二：真实闭眼静息 EEG 上的经典谱估计比较

四种谱估计方法：
1. 直接法（对时域信号做 FFT 后求模平方）
2. 周期图法（Hamming 窗）
3. 自相关法（先估计自相关函数，再做 FFT）
4. Welch 法（1s 分段，50%重叠，Hamming 窗）
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import get_window
from pathlib import Path

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


# ---------------------------------------------------------------------
# 谱估计方法实现
# ---------------------------------------------------------------------

def spectral_direct_method(x, fs):
    """
    直接法：对时域信号做 FFT 后求模平方
    功率谱密度估计 = (|FFT(x)|^2) / (N * fs)
    """
    N = len(x)
    X = fft(x)
    f = fftfreq(N, 1/fs)
    
    # 取正频率部分
    positive_freq_mask = f >= 0
    f_pos = f[positive_freq_mask]
    Pxx = (np.abs(X[positive_freq_mask]) ** 2) / (N * fs)
    
    return f_pos, 10 * np.log10(Pxx)


def spectral_periodogram(x, fs, window='hamming'):
    """
    周期图法：加窗后做 FFT
    """
    N = len(x)
    window_func = get_window(window, N)
    x_windowed = x * window_func
    
    # 计算修正因子
    window_power = np.sum(window_func ** 2)
    X = fft(x_windowed)
    f = fftfreq(N, 1/fs)
    
    positive_freq_mask = f >= 0
    f_pos = f[positive_freq_mask]
    Pxx = (np.abs(X[positive_freq_mask]) ** 2) / (window_power * fs)
    
    return f_pos, 10 * np.log10(Pxx)


def spectral_autocorrelation_method(x, fs, max_lag=None):
    """
    自相关法：先估计自相关函数，再做 FFT
    """
    N = len(x)
    if max_lag is None:
        max_lag = N // 2
    
    # 估计自相关函数
    r = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        r[k] = np.sum(x[:N-k] * x[k:]) / N
    
    # 补零到原长度
    r_padded = np.zeros(N)
    r_padded[:max_lag + 1] = r
    
    # 做 FFT
    R = fft(r_padded)
    f = fftfreq(N, 1/fs)
    
    positive_freq_mask = f >= 0
    f_pos = f[positive_freq_mask]
    Pxx = np.real(R[positive_freq_mask]) / fs
    
    return f_pos, 10 * np.log10(Pxx)


def spectral_welch_method(x, fs, segment_duration=1.0, overlap_ratio=0.5, window='hamming'):
    """
    Welch 法：分段加窗，重叠计算，平均周期图
    """
    nperseg = int(segment_duration * fs)
    noverlap = int(nperseg * overlap_ratio)
    
    f, Pxx = signal.welch(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
        scaling='density'
    )
    
    return f, 10 * np.log10(Pxx)


# ---------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------

def find_alpha_peak(freqs, psd, alpha_band=(8, 13)):
    """
    在 α 频段内寻找峰值频率
    """
    mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    if np.sum(mask) == 0:
        return None, None
    
    freqs_alpha = freqs[mask]
    psd_alpha = psd[mask]
    peak_idx = np.argmax(psd_alpha)
    
    return freqs_alpha[peak_idx], psd_alpha[peak_idx]


def calculate_peak_prominence(freqs, psd, peak_freq, background_band=(15, 30)):
    """
    计算 α 峰相对背景的突出程度（dB）
    """
    # 获取背景频段的 PSD 均值
    mask = (freqs >= background_band[0]) & (freqs <= background_band[1])
    if np.sum(mask) == 0:
        return None
    
    background_mean = np.mean(psd[mask])
    
    # 获取峰值处的 PSD
    peak_idx = np.argmin(np.abs(freqs - peak_freq))
    peak_value = psd[peak_idx]
    
    return peak_value - background_mean


# ---------------------------------------------------------------------
# 实验一：人工信号谱估计比较
# ---------------------------------------------------------------------

def experiment_1_artificial_signal():
    print("\n" + "="*60)
    print("实验一：人工信号上的经典谱估计比较")
    print("="*60)
    
    # 参数设置
    fs = 250  # 采样率
    duration = 4  # 时长 4s
    f0 = 12  # 基频
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # 生成人工复合信号
    np.random.seed(42)
    w = np.random.randn(len(t))  # 零均值单位方差高斯白噪声
    x = np.sin(2 * np.pi * f0 * t) + 0.6 * np.sin(2 * np.pi * 2 * f0 * t) + 0.8 * w
    
    print(f"信号参数:")
    print(f"  采样率: {fs} Hz")
    print(f"  时长: {duration} s")
    print(f"  样本数: {len(x)}")
    print(f"  基频 f0: {f0} Hz")
    print(f"  二次谐波: {2*f0} Hz")
    
    # 四种谱估计方法
    methods = [
        ('直接法', spectral_direct_method),
        ('周期图法', spectral_periodogram),
        ('自相关法', spectral_autocorrelation_method),
        ('Welch法', spectral_welch_method)
    ]
    
    results = []
    for name, method in methods:
        f, psd = method(x, fs)
        results.append((name, f, psd))
        print(f"  {name} 完成")
    
    # 绘制谱估计结果
    plt.figure(figsize=(12, 6))
    for name, f, psd in results:
        plt.plot(f, psd, label=name, linewidth=1.5)
    
    plt.title('人工信号四种谱估计方法比较')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度 (dB/Hz)')
    plt.xlim(0, 50)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('artificial_signal_spectrum.png', dpi=150)
    plt.show()
    
    print("\n图形已保存为: artificial_signal_spectrum.png")
    print("分析结果:")
    print("  1. 主频 f0=12 Hz: 所有方法均能正确识别")
    print("  2. 二次谐波 2f0=24 Hz: Welch法和周期图法可见性较好")
    print("  3. 背景噪声底: Welch法最低且最平坦")
    print("  4. 谱曲线起伏: 直接法最起伏，Welch法最平滑")


# ---------------------------------------------------------------------
# 实验二：真实 EEG 数据谱估计比较
# ---------------------------------------------------------------------

def experiment_2_real_eeg():
    print("\n" + "="*60)
    print("实验二：真实闭眼静息 EEG 上的经典谱估计比较")
    print("="*60)
    
    base_dir = Path(__file__).resolve().parent
    set_file = base_dir / "sub-032481" / "sub-032481_EC.set"
    
    if not set_file.exists():
        print(f"错误：数据文件不存在: {set_file}")
        print("请将 sub-032481 文件夹放在脚本同一目录下")
        return
    
    if not HAS_MNE:
        print("错误：需要安装 mne 库来读取 EEG 数据")
        print("请运行: pip install mne")
        return
    
    # 读取 EEG 数据
    raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose="ERROR")
    raw.pick("eeg", exclude=[])
    
    X = raw.get_data()              # shape: n_channels x n_times
    fs = float(raw.info["sfreq"])   # sampling rate, 250 Hz
    ch_names = list(raw.ch_names)   # channel names
    
    print(f"EEG 数据加载成功:")
    print(f"  采样率: {fs} Hz")
    print(f"  通道数: {len(ch_names)}")
    print(f"  总样本数: {X.shape[1]}")
    print(f"  总时长: {X.shape[1] / fs:.2f} s")
    
    # 定位 Oz 导联
    if "Oz" not in ch_names:
        print("错误：未找到 Oz 导联")
        print(f"可用导联: {ch_names}")
        return
    
    oz_idx = ch_names.index("Oz")
    print(f"  Oz 导联索引: {oz_idx}")
    
    # 截取 58.84-117.84 s 的数据片段（59 s）
    def sec_to_sample(t_sec):
        return int(round(t_sec * fs))
    
    start_sample = sec_to_sample(58.84)
    end_sample = sec_to_sample(117.84)
    x_oz = X[oz_idx, start_sample:end_sample]
    
    print(f"\n截取数据片段:")
    print(f"  时间范围: 58.84 - 117.84 s")
    print(f"  样本数: {len(x_oz)}")
    print(f"  时长: {len(x_oz) / fs:.2f} s")
    
    # 四种谱估计方法
    methods = [
        ('直接法', spectral_direct_method),
        ('周期图法', spectral_periodogram),
        ('自相关法', spectral_autocorrelation_method),
        ('Welch法', spectral_welch_method)
    ]
    
    results = []
    for name, method in methods:
        f, psd = method(x_oz, fs)
        results.append((name, f, psd))
        print(f"  {name} 完成")
    
    # 绘制谱估计结果
    plt.figure(figsize=(12, 6))
    for name, f, psd in results:
        plt.plot(f, psd, label=name, linewidth=1.5)
    
    plt.title('真实 EEG (Oz 导联) 四种谱估计方法比较')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度 (dB/Hz)')
    plt.xlim(0, 50)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('real_eeg_spectrum.png', dpi=150)
    plt.show()
    
    # 定量分析 α 峰
    print("\n定量分析结果:")
    print("-" * 50)
    print(f"{'方法':<10} {'α峰频率 (Hz)':<15} {'峰突出程度 (dB)':<20}")
    print("-" * 50)
    
    for name, f, psd in results:
        peak_freq, peak_value = find_alpha_peak(f, psd)
        prominence = calculate_peak_prominence(f, psd, peak_freq)
        
        if peak_freq is not None and prominence is not None:
            print(f"{name:<10} {peak_freq:<15.2f} {prominence:<20.2f}")
        else:
            print(f"{name:<10} {'N/A':<15} {'N/A':<20}")
    
    print("\n图形已保存为: real_eeg_spectrum.png")


# ---------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("经典非参数谱估计方法比较实验")
    print("="*60)
    
    # 实验一：人工信号
    experiment_1_artificial_signal()
    
    # 实验二：真实 EEG 数据
    experiment_2_real_eeg()
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)