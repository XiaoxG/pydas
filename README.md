# SKLOE PyDAS

A python package for experiment data analysis (FDAS in Python!).

Copyright &copy; 2018

**Contributors:**
* Zhichao Fang
* Xiaoxian Guo

## 使用方法
*或：*

1. `git clone https://github.com/XiaoxG/pydas.git`至项目目录
2. `import CaseData`

*或：*

1. `git clone https://github.com/XiaoxG/pydas.git` 至本地目录
2. 在site-packages添加一个路径文件，如pydas.pth，必须以.pth为后缀，写上保存 pydas.py 的本地所在的目录名称。
3. 在编写python脚本时，使用 `from CaseData import *`

**Get Started**: example.ipynb

**注意事项：**

1. 找不到python系统目录，可以使用以下命令查看系统python安装目录：

```python
import sys
sys.path
```

## 简介

基于python3.6环境开发，主要针对FDAS处理后的数据，进行进一步后处理，实现数据读入，数据处理，数据输出。

FDAS主要对单个工况数据进行处理，我们希望基于python开发流程化的后处理工具，提高工作效率，并提供python接口，可以实现更多的实时功能。

主要应用场景：

- 不同工况，同一通道的数据横向对比分析
- 标准化的批量报表自动生成
- 标准化流程化的数据后处理
- 便捷的自定义数据处理
- 应用于自动校波、校风

...

目前程序托管于GitHub，主页为：https://github.com/XiaoxG/pydas

目前拥有以下功能：

1. *.out 文件读入与保存
2. 打印文件基本信息到显示屏，txt文本，excel文本 （包括：段数，通道数，采样频率，采样时间，前标定段信息，各个通道单位，通道名，系数，每段数据的采样点数，起止时间等）
3. 打印基本统计信息到显示屏，txt文本，excel文本 （包括：最大值，最小值，方差，均值，点数等）
4. 输出数据至*.mat文件或*.dat文件
5. 通道单位修复 （原out文件通道单位不能超过4个字符）
6. 实型值转换
7. 数据时域统计分析（高低频分离）（基于Pandas）
8. 校波文件读入与数据对其
9. 运动数据与加速度自动对其
10. 高低频滤波、取平均值

系统要求：
- Python >= 3.6.3
- Pandas >= 0.20.3
- Matplotlib >= 2.1.0
- Scipy >= 0.19.1
- PyFFTW >= 0.10.1（用于fft1d）

Windows下强烈推荐使用[Anaconda](https://www.anaconda.com)构建Python科学计算环境。

读入数据后，数据类型为pandas.DataFrame，基于pandas库的各种函数均可调用，请发挥你的想象力。

Pandas documents: https://pandas.pydata.org

后续处理可以应用pywafo开发统计处理模块，包括时域分析，频谱转换，极值预报，统计模型检验，疲劳分析等。

强烈推荐WAFO模块（Matlab, Python）: http://www.maths.lth.se/matstat/wafo/

更多功能会陆续推出，欢迎大家在Github平台参与代码开发

如有任何问题请联系：[xiaoxguo@sjtu.edu.cn](mailto:xiaoxguo@sjtu.edu.cn)
