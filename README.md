# Brain Tumor Detection using EfficientNet-B0

![Brain MRI](https://user-images.githubusercontent.com/placeholder/brain_mri_example.png)

A deep learning project to **detect brain tumors from MRI scans** using **transfer learning** with **EfficientNet-B0**. The project includes:

* Dataset exploration and visualization
* Preprocessing and data augmentation
* Model training and evaluation
* Prediction on new images
* Grad-CAM for visual interpretability

---

## Table of Contents

* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Data Preprocessing](#data-preprocessing)
* [Model Architecture](#model-architecture)
* [Training & Evaluation](#training--evaluation)
* [Prediction](#prediction)
* [Grad-CAM Visualizations](#grad-cam-visualizations)
* [Saving and Loading Model](#saving-and-loading-model)
* [Results](#results)
* [Future Improvements](#future-improvements)

---

## Dataset

The project uses a **brain MRI dataset** with the following structure:

```
brain_mri_dataset/
└── brain_tumor_dataset/
    ├── yes/   # Tumor present
    └── no/    # No tumor
```

* **Number of images**: ~253
* **Classes**: `"yes"` (tumor), `"no"` (healthy)

> The dataset can be downloaded from Kaggle: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

---

## Project Structure

```
brain-tumor-detection/
├── data/                   # Raw dataset
├── model/                  # Saved model weights
├── notebooks/              # EDA and experiments
├── src/                    # Scripts for training, prediction, Grad-CAM
├── README.md
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

2. Install dependencies:

```bash
pip install torch torchvision matplotlib numpy scikit-learn pillow tqdm opencv-python seaborn
```

> Recommended: Use **Python 3.9+** and a GPU for faster training.

---

## Data Preprocessing

* Resize images to **224x224**
* Apply **data augmentation** on training images:

  * Random horizontal flips
  * Random rotations
* Normalize using **ImageNet mean and standard deviation**:

```python
[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
```

* Train/Validation split: **80% / 20%**, stratified by class

---

## Model Architecture

* **Base Model**: EfficientNet-B0 pretrained on ImageNet
* **Classifier Layer** replaced to output 2 classes (`yes`, `no`)

```python
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
```

* Loss: **Weighted Cross-Entropy** to handle class imbalance
* Optimizer: **AdamW** with learning rate `1e-4`

---

## Training & Evaluation

* **Epochs**: 10
* **Batch size**: 32
* Device: CPU or GPU

```python
for epoch in range(EPOCHS):
    train_one_epoch(model, train_loader)
    train_acc, _, _ = evaluate(model, train_loader)
    val_acc, _, _ = evaluate(model, val_loader)
```

* **Validation accuracy plateaued** around **96%**
* Visualized training/validation accuracy over epochs:

```python
plt.plot(train_acc_hist, marker='o', label="Train")
plt.plot(val_acc_hist, marker='s', label="Validation")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

---

## Prediction

Predict brain tumor from a new MRI image:

```python
from src.predict import predict_image

result = predict_image("brain_mri_dataset/brain_tumor_dataset/yes/Y123.jpg", model, device=DEVICE)
print(result)
# {'prediction': 'yes', 'confidence': 0.95}
```

---

## Grad-CAM Visualizations

* Grad-CAM highlights **regions influencing the model's decision**

```python
from src.gradcam import GradCAM

cam = GradCAM(model)
img, _ = val_ds[0]
input_img = img.unsqueeze(0).to(DEVICE)
heatmap = cam.generate(input_img)[0].cpu().numpy()

plt.imshow(heatmap, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")
plt.show()
```

* Bright regions indicate **areas the model focused on**, e.g., the tumor

Optional overlay on the original MRI:

```python
overlay = 0.5*heatmap_color + 0.5*img_np
plt.imshow(overlay)
```

---

## Saving and Loading Model

```python
# Save model
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes,
    "arch": "efficientnet_b0"
}, "model/brain_mri_efficientnet_b0.pth")

# Load model
checkpoint = torch.load("model/brain_mri_efficientnet_b0.pth", map_location="cpu")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(checkpoint["classes"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
classes = checkpoint["classes"]
```

---

## Results

* **Validation Accuracy**: ~96%
* **Precision / Recall / F1 Score**: High for both classes (especially tumor detection)
* **Grad-CAM** confirms model focuses on tumor regions

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| yes   | 0.96      | 0.95   | 0.96     |
| no    | 0.97      | 0.97   | 0.97     |

* **Visual inspection** with `show_predictions()` shows correct predictions highlighted in green and mispredictions in red

---

## Future Improvements

* Expand dataset for **more robust training**
* Experiment with **other architectures** (EfficientNet-B3, ResNet50)
* Deploy as **Streamlit / Flask app** for real-time MRI prediction
* Include **uncertainty quantification** for medical-grade decision support

---

## References

* [Brain MRI Dataset on Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
* [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
* [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)

---

### Demo

```python
from src.visualize import show_predictions

show_predictions(model, val_ds, n=6)
```

* Displays **random validation images** with true vs predicted labels
* Color-coded for **easy inspection**

---

**Project by David Obi**
