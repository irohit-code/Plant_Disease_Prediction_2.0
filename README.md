🌿 Plant Disease Detection Using Deep Learning

This project is an end-to-end system to detect plant leaf diseases using deep learning (ResNet50), with a professional GUI for easy usage. Users can upload a leaf image and get the predicted disease along with a confidence score.

📚 Dataset

Dataset used: PlantVillage Dataset

Description:

The dataset contains images of leaves from various plants with different diseases.

Each class corresponds to a specific disease or healthy leaf.

Images are in color format.

For this project, we used the color images.

Structure of dataset:

data/
    color/
        Tomato_healthy/
        Tomato_Leaf_Mold/
        Potato_Early_Blight/
        ...


Split:

train/ — used for training the model

test/ — used for evaluating the model performance

📂 Project Files
File/Notebook	Description
01_dataloader.ipynb	Loads dataset, applies transforms, and creates DataLoader for training and testing. Supports GPU acceleration.
02_transforms.ipynb	Applies image preprocessing and augmentation (resize, normalization, flip, etc.) and visualizes sample images.
03_model_resnet.ipynb	Implements ResNet50 with a custom classifier head for number of classes in the dataset.
04_train.ipynb	Trains the model on GPU, saves checkpoints after every epoch, logs loss and accuracy. Can resume from last checkpoint.
05_evaluate.ipynb	Evaluates model performance on test data. Generates accuracy, confusion matrix, etc.
06_deployment_gui.py	Standalone GUI using CustomTkinter. Allows:
• Uploading a leaf image
• Displaying image
• Checking disease prediction
• Viewing confidence score
models/best_resnet50.pth	Trained model weights saved after training.
README.md	Project documentation (this file).
🛠 Tools & Libraries Used

Python 3.x

PyTorch — For deep learning and GPU acceleration

torchvision — Pretrained models, transforms, and datasets

PIL / Pillow — For image processing

CustomTkinter — Modern, professional GUI

Tkinter — File dialog and GUI backend

Pathlib — For handling file paths

💻 Steps to Run the Project
1. Clone the repository
git clone <your-repo-url>
cd Plant-Disease-Detection

2. Install required libraries
pip install torch torchvision pillow customtkinter

3. Place dataset

Download PlantVillage color dataset.

Organize into ../data/splits/train and ../data/splits/test folders.

4. Training the model (optional)

Open 04_train.ipynb

Run all cells to train ResNet50 on the dataset.

Checkpoints will be saved in the models/ folder.

Note: Pretrained weights (best_resnet50.pth) are already included for deployment.

5. Evaluate the model

Open 05_evaluate.ipynb

Run to see accuracy, confusion matrix, and performance metrics.

6. Run the GUI

Open or run 06_deployment_gui.py

A window will pop up with:

Upload Leaf Image button — select a leaf image

Check Status button — predict disease

Confidence bar — shows prediction confidence

Right panel — displays uploaded image and predicted disease

🧠 Model Details

Architecture: ResNet50 pretrained backbone + custom classifier head

Classifier head:

Linear(2048 -> 512) → ReLU → Dropout(0.3) → Linear(512 -> num_classes)


Loss function: Cross-entropy

Optimizer: Adam

Training: GPU-enabled, checkpoints saved every epoch

Evaluation metrics: Accuracy, confusion matrix

🎨 GUI Details

Library: CustomTkinter

Features:

Modern dark theme

Split layout: Left panel (buttons + confidence), Right panel (image + prediction)

Uploaded image display

Prediction with clean class names (underscores removed, capitalized)

Confidence displayed as progress bar

✅ Project Highlights

End-to-end leaf disease detection from dataset to GUI

GPU-accelerated training with checkpointing

Professional standalone desktop GUI for non-coders

Easily extendable for more plant diseases

📌 Notes

For best performance, use a machine with a GPU.

Make sure the dataset folder structure matches the expected train / test split.

You can extend the GUI to include real-time webcam detection in future.

This README ensures that even someone with zero coding experience can:

Understand what the project does

Know the dataset and files

Run the model and GUI

Use the system to predict plant diseases
