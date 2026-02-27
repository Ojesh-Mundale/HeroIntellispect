from ultralytics import YOLO

model = YOLO("best.pt")

def detect_damage(image_path):
    results = model(image_path)
    boxes = results[0].boxes.cls.tolist()
    names = model.names
    
    damage_list = [names[int(cls)] for cls in boxes]
    
    return damage_list, results
