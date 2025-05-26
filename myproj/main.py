from ultralytics import YOLO

model = YOLO("yolo11s.pt")

result = model.train(
    data="/ultralytics/myproj/MyProj4/data.yaml",
    epochs=100,
    imgsz=640,
    augment=False,
    batch=16
)

