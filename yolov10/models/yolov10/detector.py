import cv2
from PIL import Image
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def inference(image_path, weight_path):
    model = YOLO(weight_path)
    pil_img = Image.open(image_path)
    result = model(source=pil_img, conf=0.5,
                verbose=False)[0]
    result_img = result.plot()
    result_img = cv2.cvtColor(result_img, 
                            cv2.COLOR_BGR2RGB)

    return result_img

def inference_sahi(image_path, weight_path):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=weight_path,
        confidence_threshold=0.5,
        device="cpu",  # or 'cuda:0'
    )
    

    result = get_sliced_prediction(
        image_path,
        detection_model=detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    return result