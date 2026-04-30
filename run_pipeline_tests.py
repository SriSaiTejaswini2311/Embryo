import os
import numpy as np
from PIL import Image
import cv2

from predict_validator import EmbryoValidator
from predict_module import EmbryoClassifier
from predict_malpani import EmbryoPredictor

validator = EmbryoValidator(model_path="embryo_validator_model.h5")
classifier = EmbryoClassifier(model_path="embryo_model_turbo.h5")
grader = EmbryoPredictor(model_path="embryo_grading_v4.pth")

def run_pipeline(image_path):
    results = {
        'status': 'Processing',
        'is_valid': None,
        'val_confidence': None,
        'stage': None,
        'stage_confidence': None,
        'grading': None,
        'human_review_required': False,
        'error': None
    }
    
    try:
        is_valid, val_conf = validator.validate(image_path)
        results['is_valid'] = is_valid
        results['val_confidence'] = val_conf
        
        if not is_valid:
            results['status'] = 'Rejected by Validator'
            return results
            
        stage, stage_conf, _ = classifier.predict(image_path)
        results['stage'] = stage
        results['stage_confidence'] = stage_conf
        
        if stage_conf < 0.70:
            results['human_review_required'] = True
            
        if stage == "Blastocyst":
            full_grade, grading_res = grader.predict(image_path)
            results['grading'] = grading_res
            if grading_res and grading_res.get('low_confidence', False):
                 results['human_review_required'] = True
                 
        results['status'] = 'Success'
        
    except Exception as e:
        results['status'] = 'Error'
        results['error'] = str(e)
        
    return results

def print_results(test_name, res):
    print(f"\n--- {test_name} ---")
    for k, v in res.items():
        if v is not None:
            print(f"{k}: {v}")

os.makedirs("test_cases", exist_ok=True)
cv2.imwrite("test_cases/blank.jpg", np.zeros((160, 160, 3), dtype=np.uint8))
with open("test_cases/corrupted.jpg", "w") as f:
    f.write("This is not a real image file, just text bytes simulating corruption.")
cv2.imwrite("test_cases/noisy.jpg", np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8))

res1 = run_pipeline("embryo_ai_logo_1776785134392.png")
print_results("TEST 1: Invalid Image (Logo)", res1)

res2 = run_pipeline("2b.jpeg")
print_results("TEST 2: Valid Embryo (2b.jpeg)", res2)

res_blank = run_pipeline("test_cases/blank.jpg")
print_results("TEST 4A: Blank Image", res_blank)

res_corrupt = run_pipeline("test_cases/corrupted.jpg")
print_results("TEST 4B: Corrupted File", res_corrupt)

res_noisy = run_pipeline("test_cases/noisy.jpg")
print_results("TEST 4C: Extremely Noisy Image", res_noisy)
