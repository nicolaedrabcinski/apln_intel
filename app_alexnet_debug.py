from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader  # type: ignore
from torchvision import transforms, models  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import os
import warnings
import streamlit as st  # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning, message="Can't initialize NVML")


# Set title
st.title("Intel Dataset Image Classification")

# Define class Dataset
class ImageFolderCustom(Dataset):
    def __init__(self, target_dir, transform):
        self.paths = list(Path(target_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = class_finder(target_dir)

        # Debugging print statement
        print(f"Found {len(self.paths)} images in {target_dir}")
        print(f"Classes found: {self.classes}")

    def load_image(self, index):
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, indx):
        img = self.load_image(indx)
        class_name = self.paths[indx].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            img = self.transform(img)
            # Debugging print statement
            print(f"Loaded image at index {indx}: {img.shape}, Class: {class_name} ({class_idx})")
            return img, class_idx
        else:
            return img, class_idx

# Function for class finder
def class_finder(directory):
    classes = sorted(i.name for i in os.scandir(directory) if i.is_dir())
    if not classes:
        raise FileNotFoundError(f'This directory does not have any classes: {directory}')

    class_to_idx = {name: value for value, name in enumerate(classes)}
    return classes, class_to_idx

# Define transforms
test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.425, 0.415, 0.405), (0.255, 0.245, 0.235))
])

# Load validation dataset
image_folder = "./data/seg_test/seg_test"  # Replace with your validation data folder
dataset = ImageFolderCustom(image_folder, test_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Debugging print statement for the number of batches
print(f"Total batches in the dataloader: {len(dataloader)}")

# Load the model architecture
model_architecture = models.alexnet(pretrained=False)  # Define the architecture of your model
model_architecture.classifier[6] = nn.Linear(model_architecture.classifier[6].in_features, len(dataset.classes))  # Adjust the final layer

# Load the specific model file
model_file = Path("/home/viorel/apln_intel_classification/models/AlexNet_full.pt")
model_state_dict = torch.load(model_file)  # Load the state dictionary

# Debugging print statement to check model loading
print(f"Loaded model file: {model_file.name}")
model_architecture = torch.load(model_file)  # Load model into memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_architecture = model_architecture.to(device)

# Display selected model name in Streamlit
st.write(f"Using model: {model_file.name}")

# Move model to appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_architecture = model_architecture.to(device)

# Variables to store predictions
y_true = []
y_pred = []

# Run inference
with torch.no_grad():
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Debugging print statement to check input shape
        print(f"Batch inputs shape: {inputs.shape}, Batch labels shape: {labels.shape}")

        outputs = model_architecture(inputs)
        
        # Debugging print statement to check model outputs
        print(f"Outputs shape: {outputs.shape}")
        
        _, preds = torch.max(outputs, 1)
        
        # Debugging print statement for predictions
        print(f"Predictions: {preds.cpu().numpy()}")

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Check if predictions were made
if len(y_true) > 0 and len(y_pred) > 0:
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Display metrics
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1-Score: {f1:.4f}")
else:
    st.write("No predictions made or no data available.")

