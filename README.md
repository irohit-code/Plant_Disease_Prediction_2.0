# Plant Disease Detection Using Deep Learning

This project is an end-to-end system to detect plant leaf diseases using deep learning (ResNet50), with a professional GUI for easy usage. Users can upload a leaf image and get the predicted disease along with a confidence score.

---

## Project Structure

```
Plant-Disease-Detection/
│
├── data/                   # Dataset folder
│   ├── splits/             # Train/Test split
│   │   ├── train/          # Training images organized in class folders
│   │   └── test/           # Testing images organized in class folders
│   └── color/              # Full color dataset (optional)
│
├── models/                 # Saved model weights
│   └── best_resnet50.pth   # Trained model
│
├── notebooks/              # Jupyter notebooks
│   ├── 01_dataloader.ipynb # Load dataset and create DataLoader
│   ├── 02_transforms.ipynb # Preprocessing and augmentation
│   ├── 03_model_resnet.ipynb # Define ResNet50 model with classifier
│   ├── 04_train.ipynb      # Training loop with GPU support and checkpointing
│   └── 05_evaluate.ipynb   # Evaluation metrics and confusion matrix
│
├── 06_deployment_gui.py    # Standalone GUI for leaf disease prediction
├── README.md               # Project documentation (this file)
└── requirements.txt        # Required Python libraries (optional)
```

---

## Dataset

**Dataset used:** PlantVillage Dataset (Color Images)

- Contains images of leaves from various plants with different diseases.
- Each class corresponds to a specific disease or healthy leaf.
- Images used in this project are in color format.
- Only the color images folder is used.

**Dataset structure:**
```
data/
    color/
        Tomato_healthy/
        Tomato_Leaf_Mold/
        Potato_Early_Blight/
        ...
```

**Split:**
- `train/` — used for training the model
- `test/` — used for evaluating model performance

---

## Tools & Libraries

- Python 3.x
- PyTorch
- torchvision
- PIL / Pillow
- CustomTkinter
- Tkinter
- Pathlib

---

## Steps to Run the Project

### 1. Clone the repository
```
git clone <your-repo-url>
cd Plant-Disease-Detection
```

### 2. Install required libraries
```
pip install torch torchvision pillow customtkinter
```

### 3. Place dataset
- Download PlantVillage color dataset.
- Organize into `../data/splits/train` and `../data/splits/test` folders.

### 4. Training the model (optional)
- Open `04_train.ipynb`
- Run all cells to train ResNet50 on the dataset.
- Checkpoints will be saved in the `models/` folder.

> Pretrained weights (`best_resnet50.pth`) are included for deployment.

### 5. Evaluate the model
- Open `05_evaluate.ipynb`
- Run to see accuracy, confusion matrix, and performance metrics.

### 6. Run the GUI
- Open or run `06_deployment_gui.py`
- A window will pop up with:
  - **Upload Leaf Image button** — select a leaf image
  - **Check Status button** — predict disease
  - **Confidence bar** — shows prediction confidence
  - **Right panel** — displays uploaded image and predicted disease

---

## Model Details

- Architecture: ResNet50 pretrained backbone + custom classifier head
- Classifier head:
```
Linear(2048 -> 512) → ReLU → Dropout(0.3) → Linear(512 -> num_classes)
```
- Loss function: Cross-entropy
- Optimizer: Adam
- Training: GPU-enabled, checkpoints saved every epoch
- Evaluation metrics: Accuracy, confusion matrix

---

## GUI Details

- Library: CustomTkinter
- Features:
  - Modern dark theme
  - Split layout: Left panel (buttons + confidence), Right panel (image + prediction)
  - Uploaded image display
  - Prediction with clean class names (underscores removed, capitalized)
  - Confidence displayed as progress bar

---

## Project Highlights

- End-to-end leaf disease detection from dataset to GUI
- GPU-accelerated training with checkpointing
- Professional standalone desktop GUI for non-coders
- Easily extendable for more plant diseases

---

## Notes

- For best performance, use a machine with a GPU.
- Make sure the dataset folder structure matches the expected train/test split.
- The GUI can be extended in future to include real-time webcam detection.

---

This project is designed so that even non-coders can understand and use it by following the instructions above.

