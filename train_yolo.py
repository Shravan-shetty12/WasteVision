from ultralytics import YOLO

def main():
    DATA_YAML_PATH = "dataset/data.yaml"

    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_YAML_PATH,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0
    )

    model.save("trained_model.pt")
    print("Training complete. Model saved as trained_model.pt")

if __name__ == "__main__":
    main()
