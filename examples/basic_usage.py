#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic usage example for PyDAS with simplified directory structure
"""

# 以下两种导入方式在安装后均可使用
# 方式1: 直接导入PyDAS类
from pydas import PyDAS

# 方式2: 导入waveModel子包 
import pydas.waveModel as wm

import numpy as np
import matplotlib.pyplot as plt

# 创建合成数据用于测试
def create_synthetic_data(filename="synthetic_data.csv"):
    """Create synthetic data for testing"""
    # 时间向量: 20秒，采样率10Hz
    fs = 10.0  # Hz
    t = np.arange(0, 20, 1/fs)
    
    # 创建包含多个频率成分的信号
    f1, f2, f3 = 0.5, 1.0, 2.0  # Hz
    a1, a2, a3 = 1.0, 0.5, 0.25  # 振幅
    
    # 具有三个谱峰的信号
    signal = a1 * np.sin(2 * np.pi * f1 * t) + \
             a2 * np.sin(2 * np.pi * f2 * t) + \
             a3 * np.sin(2 * np.pi * f3 * t) + \
             0.1 * np.random.randn(len(t))  # 添加噪声
    
    # 创建DataFrame
    import pandas as pd
    df = pd.DataFrame({
        'Time': t,
        'signal': signal,
        'signal2': signal * 0.8 + 0.2 * np.random.randn(len(t))
    })
    
    # 保存为CSV文件
    df.to_csv(filename, index=False)
    
    return filename

def main():
    """主示例函数"""
    print("PyDAS 基本使用示例")
    print("------------------")
    
    # 创建合成测试数据
    test_file = create_synthetic_data()
    print(f"创建合成数据文件: {test_file}")
    
    # 使用PyDAS加载数据
    data = PyDAS(filename=test_file)
    print(f"加载数据，共{len(data.channels)}个通道")
    print(f"可用通道: {data.channels}")
    
    # 绘制通道数据
    data.plot_channel("signal", use_plotly=True, save_html="signal_plot.html")
    print("生成交互式图表: signal_plot.html")
    
    # 执行谱分析
    spec, fig = data.spectral_analysis(
        channel_name="signal",
        method="cov",
        L=128,  # 窗口大小
        use_plotly=True,
        save_html="spectrum_plot.html"
    )
    print("生成谱分析图: spectrum_plot.html")
    
    # 直接使用waveModel
    # 使用JONSWAP谱模型创建谱
    if hasattr(wm, 'jonswap'):
        print("\n使用waveModel创建JONSWAP谱")
        freq = np.linspace(0.05, 3, 100)
        Hs = 4.0  # 有义波高(米)
        Tp = 10.0  # 峰周期(秒)
        gamma = 3.3  # 峰度参数
        
        # 调用waveModel中的jonswap函数
        S = wm.jonswap(freq, Hs, Tp, gamma)
        
        # 绘制谱
        plt.figure(figsize=(10, 6))
        plt.plot(freq, S)
        plt.xlabel('频率 (Hz)')
        plt.ylabel('谱密度 (m²/Hz)')
        plt.title('JONSWAP 谱')
        plt.grid(True)
        plt.savefig('jonswap_spectrum.png', dpi=300)
        print("生成JONSWAP谱图: jonswap_spectrum.png")

if __name__ == "__main__":
    main() 