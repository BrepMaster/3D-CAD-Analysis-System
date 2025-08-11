# 3D CAD Analysis System / 3D CAD智能分析系统

This is a professional 3D CAD model classification and segmentation platform developed with PyQt5 and PythonOCC. It integrates advanced deep learning technologies (UV-Net) to perform precise classification and segmentation of STEP-format CAD models.

这是一个基于PyQt5和PythonOCC开发的专业CAD模型分类与分割平台，集成了先进的深度学习技术（UV-Net），可对STEP格式的CAD模型进行精确分类与分割。

![System Screenshot](screenshot.png)

📦 **Download (Windows EXE version)**  

链接: https://pan.baidu.com/s/1tFnZHGxATxBZq6O0OMl_VA?pwd=q8xq

提取码: q8xq

**温馨提示**  
如果本项目对您有所帮助，欢迎点击右上角 ⭐Star 支持！  
如需在学术或商业用途中使用本项目，请注明出处。

---

## Table of Contents / 目录

* [Overview / 概述](#overview--概述)
* [Key Features / 核心功能](#key-features--核心功能)
* [Usage Guide / 使用指南](#usage-guide--使用指南)
* [Project Structure / 项目结构](#project-structure--项目结构)
* [License / 许可证](#license--许可证)

---

## Overview / 概述

A comprehensive CAD analysis system with dual functionality: segmentation for identifying different face regions and classification for categorizing entire CAD models. Features intuitive 3D visualization and interactive analysis tools.

一个功能全面的CAD分析系统，具有双重功能：用于识别不同面区域的分割功能和用于分类整个CAD模型的分类功能。具有直观的3D可视化和交互式分析工具。

---

## Key Features / 核心功能

| Feature          | Description                       | 功能描述                  |
| ---------------- | --------------------------------- | --------------------- |
| Dual Mode        | Switch between segmentation/classification | 分割/分类双模式切换 |
| Bilingual UI     | Seamless EN/CN language switching | 无缝中英文界面切换 |
| Deep Learning    | UV-Net based analysis             | 基于UV-Net的分析   |
| Batch Processing | Process multiple files at once    | 批量处理功能       |
| Interactive View | Real-time 3D visualization        | 实时交互3D可视化   |     |

---


## Usage Guide / 使用指南

### Segmentation Mode / 分割模式

1. **Load Model** - Segmentation model file
   **加载模型** - 分割模型文件
2. **Load Labels** - Label mapping file
   **加载标签** - 标签映射文件
3. **Import CAD** - STEP file to analyze
   **导入模型** - 要分析的STEP文件
4. **Segment** - Run segmentation analysis
   **执行分割** - 运行分割分析
5. **View Results** - Color-coded face segmentation
   **查看结果** - 彩色编码的面分割结果

### Classification Mode / 分类模式

1. **Load Model** - Classification model file
   **加载模型** - 分类模型文件
2. **Load Labels** - Category mapping file
   **加载标签** - 类别映射文件
3. **Import CAD** - STEP file to classify
   **导入模型** - 要分类的STEP文件
4. **Classify** - Run classification
   **执行分类** - 运行分类
5. **View Results** - Category and confidence
   **查看结果** - 类别和置信度

---

## Project Structure / 项目结构

```
├── classification_model.py  # 分类模型实现
├── segmentation_model.py    # 分割模型实现  
├── graph_utils.py           # 图构建工具
├── preprocessor.py          # 数据预处理
├── segmentation_logic.py    # 分割业务逻辑
├── segmentation_ui.py       # 分割界面
├── ui_app.py                # 主应用程序
├── constants.py             # 常量和样式
└── label_config.py          # 标签配置
```

---

## License / 许可证

MIT License

---

> For technical support or commercial inquiries, please contact the development team.  
> 如需技术支持或商业合作，请联系开发团队。
