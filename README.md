
# AI PhotoSorter: Intelligent Image Quality Assessment and Organization

## 简介
AI PhotoSorter是一个先进的图片质量评估与组织工具，它利用最新的人工智能技术来自动化地管理和评估大量照片的质量。此项目源于对过去方法的回顾与现代AI技术的结合，致力于为用户提供一个高效、直观的照片管理方案。

## 主要功能
- **图片质量评估**：利用pyiqa库进行审美评估，包括清晰度、颜色饱和度、对比度等多个维度。
- **照片去重与分类**：采用lpips算法识别并排除重复照片，同时利用resnet50模型进行照片分类。
- **智能照片筛选**：基于AI评分系统，自动选取质量最高的照片，适用于社交媒体分享等场景。

## 技术亮点
- **pyiqa审美评估**：采用多模型算法，适应不同审美场景的需求。
- **LPIPS去重技术**：通过深度学习模型提取图像特征，实现高效的相似度比对。
- **ResNet50分类优化**：结合深度残差网络，提高分类的准确性和效率。

## 应用场景
AI PhotoSorter适用于个人和专业摄影师，帮助快速整理庞大的照片库，提升照片管理的效率。它特别适合那些需要在短时间内从数千张照片中筛选出最佳作品的用户。

## 快速开始
1. 克隆仓库到本地。
2. 安装所需依赖：`pip install -r requirements.txt`
3. 运行主程序：`python main.py`

## 贡献指南
欢迎对AI PhotoSorter项目做出贡献！你可以通过提交Pull Request或开设Issue来提出新功能建议或报告问题。

## 许可证
本项目遵循MIT许可证。有关详细信息，请参阅LICENSE文件。

---
AI PhotoSorter © 2023. All Rights Reserved.
