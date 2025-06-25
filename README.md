# Pneumonia Detection frmo Chest X-Rays 🫁

THis repository presents a deep learning pipeline for binary classification of chest X-Ray images into **Normal** and **Pneumonia** classes using a fine-tuned ResNet-50 model.

## 🧠 Project Objective

Develop an end-to-end deep learning model to identify pneumonia in chest X-rays, assisting medical professionals in rapid and reliable diagnosis.

## 📂 Repository Structure

- **main.ipynb**: Full pipeline for loading data, training, evaluating, and visualizing the model.
- **resnet_50.ipynb**: Implementation and fine-tuning of the ResNet-50 architecture.

## 🔢 Dataset

- **Total Images**: 5,21 chset X-ray images
- **Classes**: **Pneumonia** and **Normal**

Dataset is split into 'train', 'val' and 'test' directories.

## 🔧 Features

- 🧠 **Model**: ResNet-50 with pretrained ImageNet weights
- 🧪 **Task**: Binary Classification - Normal vs Pneumonia
- 🌀 **Transforms**: Resize, Normalize, Augmentations using `torchvision.transforms`
- 💥 **Loss Function**: CrossEntropyLoss
- 🚀 **Optimizer**: Adam or SGD with scheduler
- 📊 **Metrics**: Accuracy, Precision, Recall, F1-score
- 📉 **Visualizations**: Training & validation curves using Matplotlib

## 🛠️ Tech Stack

- Python 3
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- scikit-learn

## 📎 Notes

- THis project uses **transfer learning** - leveraging a ResNet-50 pretrained on ImageNet
- Ideal for binary medical image classification tasks
- GPU support is recommended for faster training.

---
