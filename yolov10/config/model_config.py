import os, sys

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov10'))
sys.path.append(utils_path)

class Detector_Config:
    weight_path: str = 'runs/detect/train/weights/best.pt'
    yaml_path: str = 'data/data.yaml'