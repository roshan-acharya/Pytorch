import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

st.title("🐱 Cat vs 🐟 Fish Classifier")
import os
print(os.path.exists("/tmp/model_weights.pth"))

# Define the model architecture (must match exactly)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model weights once and cache it
@st.cache_resource
def load_model():
    model = SimpleNet()
    state_dict = torch.load("./Model/model_weights.pth", map_location="cpu")  # Load weights dictionary
    model.load_state_dict(state_dict)  # Load weights into model
    model.eval()
    return model

model = load_model()

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_idx].item()

    class_names = ["Cat", "Fish"]

    st.markdown(f"### 🧠 Prediction: **{class_names[predicted_idx]}**")
    st.markdown(f"Confidence: **{confidence:.2%}**")
