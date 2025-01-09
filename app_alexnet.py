from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms, models # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import os
import warnings
import streamlit as st # type: ignore
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
            return self.transform(img), class_idx 
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
image_folder = "./data/seg_test"  # Replace with your validation data folder
dataset = ImageFolderCustom(image_folder, test_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load the model architecture
model_architecture = models.alexnet(pretrained=False)  # Define the architecture of your model
model_architecture.classifier[6] = nn.Linear(model_architecture.classifier[6].in_features, len(dataset.classes))  # Adjust the final layer

# Load the specific model file
model_file = Path("/home/viorel/apln_intel_classification/models/AlexNet_full.pt")
model_state_dict = torch.load(model_file)  # Load the state dictionary
# model_architecture.load_state_dict(model_state_dict)  # Load state_dict into the model
model_architecture = torch.load(model_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_architecture = model_architecture.to(device)

st.write(f"Loaded model: {model_file.name}")

# Select the model in Streamlit (since we only have one model)
selected_model = model_file.name

# Display selected model name
st.write(f"Using model: {selected_model}")

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
        outputs = model_architecture(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

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
