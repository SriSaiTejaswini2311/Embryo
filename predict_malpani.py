import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

# --- MODEL DEFINITION ---
class MultiHeadEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(weights='DEFAULT')
        self.base.classifier = nn.Identity()
        self.h_exp = nn.Linear(1280, 5) 
        self.h_icm = nn.Linear(1280, 4) 
        self.h_te = nn.Linear(1280, 4)
    def forward(self, x):
        f = torch.flatten(self.base(x), 1)
        return self.h_exp(f), self.h_icm(f), self.h_te(f)

class EmbryoPredictor:
    def __init__(self, model_path="embryo_grading_v4.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiHeadEfficientNet().to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded Version 4.0 Weights: {model_path}")
        else:
            print(f"⚠️ WARNING: {model_path} not found. Using untrained base.")
        
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Explicit Mappings
        self.exp_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
        self.icm_te_map = {0: 'A', 1: 'B', 2: 'C', 3: 'NA'}

    def predict(self, img_path, confidence_threshold=0.70):
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            p_exp, p_icm, p_te = self.model(tensor)
            
            # Probabilities for Auditing
            sm_exp = torch.softmax(p_exp, 1)
            sm_icm = torch.softmax(p_icm, 1)
            sm_te = torch.softmax(p_te, 1)
            
            e_idx = torch.argmax(p_exp, 1).item()
            i_idx = torch.argmax(p_icm, 1).item()
            t_idx = torch.argmax(p_te, 1).item()
            
            conf_e = sm_exp.max().item()
            conf_i = sm_icm.max().item()
            conf_t = sm_te.max().item()
            
        # Clinical Mapping
        expansion = self.exp_map[e_idx]
        icm = self.icm_te_map[i_idx]
        te = self.icm_te_map[t_idx]
        
        # Suppress Gardner string if not a full blastocyst (ICM/TE is NA)
        if icm == 'NA' or te == 'NA':
            full_grade = f"Exp {expansion} (Incomplete)"
        else:
            full_grade = f"{expansion}{icm}{te}"
            
        overall_conf = (conf_e + conf_i + conf_t) / 3
        
        result = {
            "full_grade": full_grade,
            "expansion": expansion,
            "icm": icm,
            "te": te,
            "confidence": overall_conf,
            "head_confidences": {
                "expansion": conf_e,
                "icm": conf_i,
                "te": conf_t
            },
            "low_confidence": overall_conf < confidence_threshold
        }
        
        return full_grade, result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_malpani.py <image_path>")
    else:
        predictor = EmbryoPredictor()
        grade, details = predictor.predict(sys.argv[1])
        print(f"\n🔬 V4.0 DIAGNOSTIC REPORT")
        print(f"{'='*30}")
        print(f"Final Grade: {grade}")
        if details['low_confidence']:
            print("⚠️ WARNING: Low confidence prediction. Clinical review mandatory.")
        print(f"Confidence:  {details['confidence']:.2%}")
        print(f"Head Audit:  E:{details['head_confidences']['expansion']:.2f}, I:{details['head_confidences']['icm']:.2f}, T:{details['head_confidences']['te']:.2f}")
        print(f"{'='*30}\n")
