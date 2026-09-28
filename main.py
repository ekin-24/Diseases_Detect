import io
import base64
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Start API server
app = FastAPI(title="Diseases Detect From Eye Data")
# Static path
app.mount("/static", StaticFiles(directory="static"), name="static")

# Use CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build Model
model = models.resnet50(weights=None)
num_fc = model.fc.in_features # Number of outputs
model.fc = nn.Linear(num_fc, 8) # 8 main classes

# Trained model
model.load_state_dict(torch.load("diseases_model.pth", map_location=device))
model.to(device) # Send model to device
model.eval() # Stop training only testing

# Heat map
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASS_NAMES = [
    "Normal", "Diabetic Retinopathy", "intraocular pressure", 
    "Cataract", "Macular Degeneration", "Hypertension", "Myopic", "Other Dieases"
]

# Predict
@app.post("/predict/")
async def predict_retina(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Prediction
        outputs = model(image_tensor)
        # Convert outputs to %0-%100
        probabilities = torch.sigmoid(outputs)[0] * 100
            
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                "disease": CLASS_NAMES[i],
                "probability": round(prob.item(), 2)
            })
            
        # Build heat map
        # Find the dieases with the highest probabiliti
        most_dieases = int(torch.argmax(probabilities).item())
        targets = [ClassifierOutputTarget(most_dieases)]
        
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
        
        # Convert image to numbers(0,1)
        image_array = np.array(image.resize((224, 224))) / 255.0
        
        visualization = show_cam_on_image(image_array, grayscale_cam, use_rgb=True)
        
        heatmap_pil = Image.fromarray(visualization) # Convert matris to image
        buffered = io.BytesIO()
        heatmap_pil.save(buffered, format="JPEG")    # Save the image as a JPEG in buffered
        # Encode image in base64 format
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Sort results from highest to lowest
        results.sort(key=lambda x: x["probability"], reverse=True)
        
        # Send all datas to web
        return {
            "status": "success",
            "predictions": results,
            "heatmap": heatmap_base64,
            "target_disease": CLASS_NAMES[most_dieases]
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

# In the begining show index.html
@app.get("/")
def read_root():
    return FileResponse("static/index.html")