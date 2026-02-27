from ultralytics import YOLO

model = YOLO("best.pt")

results = model("test.jpg")  # Put any vehicle image here

results[0].show()
