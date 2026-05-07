# 谱纹智瞰 - 纹理增强的高光谱叶面积指数反演软件

基于PyQt5开发的桌面应用程序，通过纹理增强技术实现高光谱图像的叶面积指数（LAI）反演。

## 功能特性

### 核心功能
- **单张图像分析**：对单张高光谱图像进行纹理特征提取和LAI反演
- **批量图像分析**：支持批量处理多张图像，提高工作效率
- **光谱预处理**：对原始光谱数据进行预处理，包括平滑、归一化等操作
- **模型构建**：基于CatBoost算法构建叶面积指数预测模型
- **模型推理**：使用训练好的模型对新图像进行LAI预测

### 技术特点
- **GLCM纹理特征提取**：计算灰度共生矩阵的多种纹理特征（对比度、相关性、能量、同质性等）
- **相关性分析**：分析纹理特征与LAI之间的相关性
- **GUI界面**：友好的图形用户界面，操作简单直观

## 技术栈

- **编程语言**：Python 3.9
- **GUI框架**：PyQt5
- **机器学习**：CatBoost
- **数据处理**：NumPy, Pandas, OpenCV
- **图像处理**：Pillow, scikit-image

## 项目结构

```
texture-lai-software/
├── code/                          # 源代码目录
│   ├── main_application.py       # 主程序入口
│   ├── main_window.py            # 主窗口界面
│   ├── batch_processing.py       # 批量处理模块
│   ├── correlation_analyzer.py   # 相关性分析模块
│   ├── glcm_calculator.py        # GLCM计算模块
│   ├── image_loader.py           # 图像加载模块
│   ├── model_inference.py        # 模型推理模块
│   ├── spectral_preprocessor.py  # 光谱预处理模块
│   ├── texture_features_extractor.py  # 纹理特征提取模块
│   └── resources.py              # 资源文件
├── data/                          # 数据目录
│   ├── GLCM.png                   # GLCM特征可视化
│   ├── GLCM.xlsx                  # GLCM特征数据
│   ├── gp.xlsx                    # 光谱数据
│   └── [其他CSV数据文件]
├── catboost_info/                 # CatBoost模型训练信息
└── README.md                      # 项目说明文档
```

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖

```
PyQt5>=5.15.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
scikit-image>=0.18.0
catboost>=1.0.0
pillow>=8.0.0
openpyxl>=3.0.0
```

## 使用说明

### 启动软件

```bash
cd code
python main_application.py
```

### 功能模块

1. **单张图像分析**
   - 加载高光谱图像
   - 提取GLCM纹理特征
   - 进行LAI反演

2. **批量图像分析**
   - 选择图像文件夹
   - 批量提取纹理特征
   - 批量进行LAI反演
   - 导出结果

3. **光谱预处理**
   - 光谱平滑处理（SG滤波）
   - 光谱归一化
   - 光谱微分

4. **模型构建**
   - 导入训练数据
   - 配置模型参数
   - 训练CatBoost模型
   - 评估模型性能

5. **模型推理**
   - 加载训练好的模型
   - 输入新图像
   - 获取LAI预测结果

## 界面预览

软件采用侧边栏导航设计，包含以下主要功能区域：
- 工具栏：功能模块切换
- 图像显示区：展示分析图像
- 结果显示区：展示反演结果和数据

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请联系项目作者。
