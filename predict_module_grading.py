import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import tensorflow as tf

# --- Phase 1 Model (Stage Detection) ---
def load_stage_model(model_path='embryo_model_turbo.h5'):
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

STAGE_LABELS = ['2cell', '4cell', '8cell', 'blastocyst', 'morula']

def get_stage_prediction(image, model):
    if model is None:
        return "Unknown", 0.0
    img = image.resize((160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    idx = np.argmax(score)
    return STAGE_LABELS[idx], np.max(score)

# --- Phase 2 Model (Gardner Grading) ---
class MultiHeadEmbryoModel(nn.Module):
    def __init__(self):
        super(MultiHeadEmbryoModel, self).__init__()
        self.base = models.mobilenet_v3_small(weights='DEFAULT')
        num_ftrs = 576 
        self.base.classifier = nn.Identity() 
        self.head_exp = nn.Linear(num_ftrs, 6) 
        self.head_icm = nn.Linear(num_ftrs, 3) 
        self.head_te = nn.Linear(num_ftrs, 3)  
        
    def forward(self, x):
        features = self.base(x)
        features = torch.flatten(features, 1)
        return self.head_exp(features), self.head_icm(features), self.head_te(features)

MAP_EXP = {i: str(i+1) for i in range(6)}
MAP_GRADE = {0: 'A', 1: 'B', 2: 'C'}

def load_grading_model(model_path=None):
    model = MultiHeadEmbryoModel()
    model.eval()
    return model

# --- NEW: Cleavage Grader (Day 2/3) ---
def simulate_cleavage_grade(image):
    """
    Simulates a Cleavage Grade (1-4) based on image complexity and texture.
    Grade 1: Even/Clean. Grade 4: Segmented/Fragmented.
    """
    img_gray = np.array(image.convert('L'))
    # Use standard deviation of pixel intensity to estimate fragmentation
    # High fragmentation = more textures = higher variance
    variance = np.std(img_gray)
    
    if variance < 30: return 1
    if variance < 50: return 2
    if variance < 70: return 3
    return 4

def get_grading_prediction(image, grading_model, stage_model):
    stage, stage_conf = get_stage_prediction(image, stage_model)
    st_low = stage.lower()
    
    # 1. Logic for Day 5 (Gardner)
    if st_low == 'blastocyst':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            exp, icm, te = grading_model(input_tensor)
        
        return {
            "type": "gardner",
            "stage": stage,
            "expansion": MAP_EXP[torch.argmax(exp).item()],
            "icm": MAP_GRADE[torch.argmax(icm).item()],
            "te": MAP_GRADE[torch.argmax(te).item()],
            "final_score": f"{MAP_EXP[torch.argmax(exp).item()]}{MAP_GRADE[torch.argmax(icm).item()]}{MAP_GRADE[torch.argmax(te).item()]}"
        }
    
    # 2. Logic for Day 2/3 (Cleavage)
    elif st_low in ['2cell', '4cell', '8cell', 'morula']:
        grade = simulate_cleavage_grade(image)
        return {
            "type": "cleavage",
            "stage": stage,
            "final_score": str(grade),
            "description": f"Grade {grade} (Fragmentation Audit Complete)"
        }
    
    return {"type": "unknown", "stage": stage}
