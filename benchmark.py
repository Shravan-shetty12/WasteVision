from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "model/best.pt"
IMAGE_PATH = "test.jpg"
IMAGE_SIZE = 640
WARMUP_RUNS = 1
BENCHMARK_RUNS = 5

print("Loading model...")
model = YOLO(MODEL_PATH)

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

print(f"Image loaded: {img.shape}")
print("Warming up...")

for _ in range(WARMUP_RUNS):
    model.predict(img, imgsz=IMAGE_SIZE, verbose=False)

print("Starting benchmark...")

times = []

for i in range(BENCHMARK_RUNS):
    start = time.perf_counter()
    model.predict(img, imgsz=IMAGE_SIZE, verbose=False)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    print(f"Run {i + 1}: {elapsed:.3f} seconds")

average = sum(times) / len(times)
print(f"\nAverage: {average:.3f} seconds")
