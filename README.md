# 🫁 Pneumonia Detection from Chest X-Rays

This repository presents a deep learning pipeline for binary classification of chest X-ray images into **Normal** and **Pneumonia** classes using both a custom Convolutional Neural Network (TensorFlow/Keras) and a fine-tuned **ResNet-50** transfer learning model (PyTorch).

---

## 🧠 Project Objective

Develop an end-to-end medical imaging pipeline capable of identifying pneumonia in chest radiographs, utilizing contrast enhancement (gamma correction and CLAHE) alongside deep convolutional architectures.

---

## 📂 Repository Structure

- `main.ipynb`: Preprocessing pipeline (grayscale conversion, gamma correction, CLAHE histogram equalization) and custom CNN training in TensorFlow/Keras.
- `resnet_50.ipynb`: Transfer learning implementation fine-tuning an ImageNet-pretrained **ResNet-50** architecture in PyTorch.
- `README.md`: Project documentation.

---

## 🔢 Dataset & Preprocessing

- **Total Images**: 5,216 chest X-ray radiographs (`train`, `val`, and `test` splits)
- **Classes**: `Normal` and `Pneumonia`
- **Image Enhancements**:
  - Look-Up Table (LUT) **Gamma Correction** ($\gamma = 0.8$)
  - **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for improved pulmonary contrast
  - Spatial resizing and normalization via `torchvision.transforms` / `OpenCV`

---

## 🛠️ Tech Stack

- **Python 3**
- **PyTorch & Torchvision** (`resnet_50.ipynb`)
- **TensorFlow & Keras** (`main.ipynb`)
- **OpenCV (`cv2`)**, **NumPy**, **scikit-learn**, **Matplotlib**

---

## 🚀 Getting Started

```bash
git clone https://github.com/sparsh0106/X-Ray-Lung-Classifier.git
cd X-Ray-Lung-Classifier
pip install torch torchvision tensorflow opencv-python numpy scikit-learn matplotlib jupyter
jupyter notebook resnet_50.ipynb
```

---

## 👤 Author

**Sparsh Agarwal** — [GitHub](https://github.com/sparsh0106) · [LinkedIn](https://www.linkedin.com/in/sparsh-agarwal-4b443b304/)
