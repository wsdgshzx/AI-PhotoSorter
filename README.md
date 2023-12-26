
[阅读中文版 README](CHINESE_README.md)

# AI PhotoSorter: Intelligent Image Quality Assessment and Organization

## Introduction
AI PhotoSorter is an advanced tool for image quality assessment and organization, utilizing the latest artificial intelligence technology to automate the management and evaluation of a large number of photos. This project is a combination of revisiting past methods and modern AI technology, committed to providing users with an efficient and intuitive photo management solution.

## Main Features
- **Image Quality Assessment**: Uses the pyiqa library for aesthetic evaluation, including dimensions such as clarity, color saturation, contrast, etc.
- **Photo Deduplication and Classification**: Identifies and eliminates duplicate photos using the lpips algorithm and classifies photos with the resnet50 model.
- **Intelligent Photo Selection**: Automatically selects the highest quality photos based on AI scoring system, suitable for social media sharing and other scenarios.

## Technical Highlights
- **pyiqa Aesthetic Assessment**: Employs multiple model algorithms to meet the needs of different aesthetic scenarios.
- **LPIPS Deduplication Technology**: Extracts image features through deep learning models for efficient similarity comparison.
- **ResNet50 Classification Optimization**: Integrates deep residual networks to enhance the accuracy and efficiency of classification.

## Application Scenarios
AI PhotoSorter is suitable for individuals and professional photographers, helping to quickly organize vast photo libraries and improve photo management efficiency. It is especially useful for users who need to select the best works from thousands of photos in a short time.

## Quick Start
1. Clone the repository to your local machine.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the main program: `python main.py`

## Contribution Guide
Contributions to the AI PhotoSorter project are welcome! You can contribute by submitting Pull Requests or opening Issues to suggest new features or report problems.

## License
This project is licensed under the MIT License. For more details, please see the LICENSE file.

---
AI PhotoSorter © 2023. All Rights Reserved.
