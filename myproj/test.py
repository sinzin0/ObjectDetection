from ultralytics import YOLO

model = YOLO("/ultralytics/myproj/runs/detect/train5/weights/best.pt")

results = model.predict(
    source="/ultralytics/myproj/MyProj/test/images",  # 테스트 이미지 폴더
    save=True,                    # 이미지 위에 결과 저장
    save_txt=True,                # 감지 결과 텍스트 저장 (YOLO format)
    conf=0.5,                 # confidence threshold
)