# 🌿 Plant Leaf Image Classification using Machine Learning

This project focuses on classifying plant leaf images using traditional machine learning algorithms (like SVM, Random Forest, and Gradient Boosting) by extracting handcrafted features such as color histograms, texture (LBP), and shape descriptors.

## 📁 Dataset

- **Source:** [Plant Pathology 2020 - FGVC7 Dataset](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7)
- **Images:** RGB images of apple leaves (healthy or affected by multiple diseases)
- **Classes:**
  - Healthy
  - Multiple Diseases
  - Rust
  - Scab

## 🧪 Objective

To classify images of plant leaves into one of the four categories using handcrafted features and traditional machine learning models instead of deep learning.

## 🧰 Technologies Used

- Python
- OpenCV, scikit-image, Mahotas – for feature extraction
- Scikit-learn – for ML modeling
- Matplotlib, Seaborn – for visualization
- Jupyter Notebook / Streamlit (optional for web interface)

## 🔍 Feature Extraction Techniques

- **Color Features:**
  - Color histograms (RGB, HSV)
- **Texture Features:**
  - Local Binary Pattern (LBP)
  - Haralick Texture Features (via Mahotas)
- **Shape Features:**
  - Contour-based shape descriptors (area, perimeter, eccentricity)

## 🤖 Models Trained

- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting Classifier
- LightGBM (optional)

## 🏆 Results

| Model              | Accuracy |
|-------------------|----------|
| SVM               | 93.5%    |
| Random Forest     | 92.1%    |
| Gradient Boosting | 94.2%    |

> Note: Accuracy may vary slightly depending on feature selection and cross-validation.

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Classification Report

## 🚀 How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/image-classification-ml.git
cd image-classification-ml
