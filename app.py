from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import time
from collections import deque, Counter

app = Flask(__name__)

# ============================================================
# WASTEVISION - WEBCAM DIAGNOSTIC / STABILITY VERSION
# ============================================================
# IMPORTANT:
# This file is based on the current app.py version, not the ZIP.
#
# The video review showed that the main problem is NOT request
# latency. The model is producing high-confidence detections on
# the background/person before the waste is clearly presented.
#
# Therefore this version:
#   1. Uses an explicit YOLO confidence threshold.
#   2. Prints EVERY raw detection to the terminal.
#   3. Prints bounding-box size/location for diagnosis.
#   4. Uses a slightly higher webcam inference size (416).
#   5. Prevents low-confidence boxes from being plotted.
#   6. Resets stale temporal state when no reliable object is seen.
#   7. Keeps confidence inside "class" so the EXISTING JS works.
#
# This improves inference handling, but it cannot repair a model
# that was trained with incorrect class mapping or insufficient
# real-world/background variation. The terminal diagnostics will
# make that problem obvious.

# ============================================================
# CLASS MAPPING
# ============================================================
CLASS_NAMES = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

# ============================================================
# INFERENCE SETTINGS
# ============================================================

# Only detections at or above this confidence are considered
# reliable by the application.
CONFIDENCE_THRESHOLD = 0.50

# Webcam inference size.
# The current JS sends 320x240 frames, so 416 improves the
# YOLO inference scale but cannot recover detail already lost
# in the browser.
WEBCAM_IMGSZ = 416

# Uploaded images can use a larger inference size.
IMAGE_IMGSZ = 640

# Temporal stabilization.
STABILITY_WINDOW = 5
STABILITY_REQUIRED = 3

# Number of consecutive frames without a reliable detection
# before clearing the displayed class.
NO_OBJECT_RESET_FRAMES = 3

# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model",
    "best.pt"
)

print("\n========================================")
print("        WASTEVISION STARTING")
print("========================================")
print("Loading YOLO model...", flush=True)
print("Model path:", MODEL_PATH, flush=True)

model = YOLO(MODEL_PATH)

print("YOLO model loaded.", flush=True)
print("Model classes:", model.names, flush=True)

# ============================================================
# MODEL WARM-UP
# ============================================================

print("Warming up YOLO model...", flush=True)

dummy_640 = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(
    source=dummy_640,
    imgsz=IMAGE_IMGSZ,
    conf=CONFIDENCE_THRESHOLD,
    verbose=False
)

dummy_416 = np.zeros((416, 416, 3), dtype=np.uint8)
model.predict(
    source=dummy_416,
    imgsz=WEBCAM_IMGSZ,
    conf=CONFIDENCE_THRESHOLD,
    verbose=False
)

print("YOLO warm-up complete.", flush=True)

# ============================================================
# WEBCAM STATE
# ============================================================
# Local single-user application: one history is sufficient.
# Unlike the previous version, missing detections clear the
# history instead of allowing an old prediction to persist.

prediction_history = deque(maxlen=STABILITY_WINDOW)

stable_class = "No object"
stable_confidence = 0.0
no_object_frames = 0


# ============================================================
# DETECTION EXTRACTION
# ============================================================

def get_detections(results, image_width, image_height):
    """
    Extract all detections and return the highest-confidence one.

    The returned diagnostic information includes:
      - class
      - class_id
      - confidence
      - bounding box
      - box width/height
      - box area percentage
      - box center
    """

    detections = []

    if results.boxes is None or len(results.boxes) == 0:
        return detections, "No object", 0.0

    for box in results.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x1 = max(0.0, min(float(x1), float(image_width)))
        y1 = max(0.0, min(float(y1), float(image_height)))
        x2 = max(0.0, min(float(x2), float(image_width)))
        y2 = max(0.0, min(float(y2), float(image_height)))

        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)

        box_area = box_width * box_height
        image_area = float(image_width * image_height)

        area_percent = (
            (box_area / image_area) * 100.0
            if image_area > 0
            else 0.0
        )

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        detections.append({
            "class": class_name,
            "class_id": class_id,
            "confidence": confidence,
            "bbox": [
                round(x1, 1),
                round(y1, 1),
                round(x2, 1),
                round(y2, 1)
            ],
            "box_width": round(box_width, 1),
            "box_height": round(box_height, 1),
            "box_area_percent": round(area_percent, 2),
            "center": [
                round(center_x, 1),
                round(center_y, 1)
            ]
        })

    best = max(
        detections,
        key=lambda item: item["confidence"]
    )

    return (
        detections,
        best["class"],
        best["confidence"]
    )


# ============================================================
# TEMPORAL STABILIZATION
# ============================================================

def stabilize_prediction(predicted_class, predicted_confidence):
    """
    Require multiple consistent frames before changing the
    displayed class.

    This is deliberately conservative:
      - low-confidence detections do not enter the history
      - no-object frames eventually clear the history
      - a class needs 3/5 recent reliable frames
    """

    global stable_class
    global stable_confidence
    global no_object_frames

    if (
        predicted_class == "No object"
        or predicted_confidence < CONFIDENCE_THRESHOLD
    ):
        no_object_frames += 1

        if no_object_frames >= NO_OBJECT_RESET_FRAMES:
            prediction_history.clear()
            stable_class = "No object"
            stable_confidence = 0.0

        return stable_class, stable_confidence

    no_object_frames = 0

    prediction_history.append(predicted_class)

    counts = Counter(prediction_history)
    candidate, votes = counts.most_common(1)[0]

    if stable_class == "No object":

        if votes >= STABILITY_REQUIRED:
            stable_class = candidate
            stable_confidence = predicted_confidence

    elif candidate != stable_class:

        if votes >= STABILITY_REQUIRED:
            stable_class = candidate
            stable_confidence = predicted_confidence

    else:

        stable_confidence = predicted_confidence

    return stable_class, stable_confidence


# ============================================================
# DISPLAY FORMATTING
# ============================================================

def format_prediction(class_name, confidence):

    if class_name == "No object":
        return "No object"

    if class_name == "Uncertain":
        return f"Uncertain ({confidence * 100:.1f}%)"

    return f"{class_name} ({confidence * 100:.1f}%)"


# ============================================================
# DEBUG LOGGING
# ============================================================

def print_detections(detections, image_width, image_height):
    """
    Print raw model detections.

    This is the most important diagnostic output for the
    current webcam accuracy problem.
    """

    if not detections:
        print("  RAW DETECTIONS: none", flush=True)
        return

    print(
        f"  RAW DETECTIONS ({image_width}x{image_height}):",
        flush=True
    )

    for index, detection in enumerate(detections, start=1):

        print(
            f"    #{index} "
            f"{detection['class']} "
            f"(id={detection['class_id']}) "
            f"conf={detection['confidence'] * 100:.1f}% "
            f"bbox={detection['bbox']} "
            f"area={detection['box_area_percent']:.1f}% "
            f"center={detection['center']}",
            flush=True
        )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# IMAGE UPLOAD
# ============================================================

@app.route("/predict_image", methods=["POST"])
def predict_image():

    total_start = time.perf_counter()

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    # -----------------------------
    # Read image
    # -----------------------------

    file_start = time.perf_counter()

    img_bytes = file.read()

    file_read_time = time.perf_counter() - file_start

    # -----------------------------
    # Decode image
    # -----------------------------

    decode_start = time.perf_counter()

    nparr = np.frombuffer(
        img_bytes,
        np.uint8
    )

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    decode_time = time.perf_counter() - decode_start

    if img is None:
        return jsonify({
            "error": "Invalid image"
        }), 400

    image_height, image_width = img.shape[:2]

    # -----------------------------
    # YOLO
    # -----------------------------

    yolo_start = time.perf_counter()

    results = model.predict(
        source=img,
        imgsz=IMAGE_IMGSZ,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )[0]

    yolo_time = time.perf_counter() - yolo_start

    # -----------------------------
    # Detections
    # -----------------------------

    (
        detections,
        predicted_class,
        predicted_confidence
    ) = get_detections(
        results,
        image_width,
        image_height
    )

    if predicted_class == "No object":
        display_class = "No object"
        display_confidence = 0.0
    else:
        display_class = predicted_class
        display_confidence = predicted_confidence

    # -----------------------------
    # Annotated image
    # -----------------------------

    plot_start = time.perf_counter()

    annotated = results.plot()

    plot_time = time.perf_counter() - plot_start

    # -----------------------------
    # JPEG
    # -----------------------------

    encode_start = time.perf_counter()

    success, buffer = cv2.imencode(
        ".jpg",
        annotated
    )

    encode_time = time.perf_counter() - encode_start

    if not success:
        return jsonify({
            "error": "Failed to encode image"
        }), 500

    encoded = base64.b64encode(
        buffer
    ).decode("utf-8")

    total_time = time.perf_counter() - total_start

    class_display = format_prediction(
        display_class,
        display_confidence
    )

    print("\n================================")
    print("IMAGE PROCESSING")
    print("================================")
    print(
        f"Image size     : {image_width}x{image_height}"
    )
    print(
        f"YOLO inference : {yolo_time:.4f} sec"
    )
    print(
        f"Plot           : {plot_time:.4f} sec"
    )
    print(
        f"JPEG encode    : {encode_time:.4f} sec"
    )
    print(
        f"TOTAL          : {total_time:.4f} sec"
    )
    print(
        f"Prediction     : {class_display}"
    )
    print("--------------------------------")

    print_detections(
        detections,
        image_width,
        image_height
    )

    print("================================\n", flush=True)

    return jsonify({
        "image": encoded,
        "class": class_display,
        "confidence": display_confidence,
        "detections": detections,
        "processing_time": total_time
    })


# ============================================================
# WEBCAM
# ============================================================

@app.route("/predict_frame", methods=["POST"])
def predict_frame():

    total_start = time.perf_counter()

    if not request.is_json:
        return jsonify({
            "error": "JSON request expected"
        }), 400

    data = request.json

    if not data or "frame" not in data:
        return jsonify({
            "error": "No frame received"
        }), 400

    frame_data = data["frame"]

    # -----------------------------
    # Decode Base64 frame
    # -----------------------------

    decode_start = time.perf_counter()

    try:

        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]

        img_bytes = base64.b64decode(
            frame_data
        )

    except Exception:
        return jsonify({
            "error": "Invalid frame data"
        }), 400

    nparr = np.frombuffer(
        img_bytes,
        np.uint8
    )

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    decode_time = time.perf_counter() - decode_start

    if img is None:
        return jsonify({
            "error": "Could not decode frame"
        }), 400

    image_height, image_width = img.shape[:2]

    # -----------------------------
    # YOLO webcam inference
    # -----------------------------

    yolo_start = time.perf_counter()

    results = model.predict(
        source=img,
        imgsz=WEBCAM_IMGSZ,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )[0]

    yolo_time = time.perf_counter() - yolo_start

    # -----------------------------
    # Extract detections
    # -----------------------------

    (
        detections,
        predicted_class,
        predicted_confidence
    ) = get_detections(
        results,
        image_width,
        image_height
    )

    # -----------------------------
    # Raw decision
    # -----------------------------

    if predicted_class == "No object":

        raw_class = "No object"

    else:

        raw_class = predicted_class

    # -----------------------------
    # Stabilized result
    # -----------------------------

    if raw_class == "No object":

        (
            final_class,
            final_confidence
        ) = stabilize_prediction(
            "No object",
            0.0
        )

    else:

        (
            final_class,
            final_confidence
        ) = stabilize_prediction(
            predicted_class,
            predicted_confidence
        )

    class_display = format_prediction(
        final_class,
        final_confidence
    )

    # -----------------------------
    # Annotate
    # -----------------------------

    plot_start = time.perf_counter()

    annotated = results.plot()

    plot_time = time.perf_counter() - plot_start

    # -----------------------------
    # JPEG
    # -----------------------------

    encode_start = time.perf_counter()

    success, buffer = cv2.imencode(
        ".jpg",
        annotated,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            60
        ]
    )

    encode_time = time.perf_counter() - encode_start

    if not success:
        return jsonify({
            "error": "Failed to encode frame"
        }), 500

    encoded = base64.b64encode(
        buffer
    ).decode("utf-8")

    total_time = time.perf_counter() - total_start

    # -----------------------------
    # DIAGNOSTIC TERMINAL OUTPUT
    # -----------------------------

    print(
        "\nWEBCAM | "
        f"Raw={predicted_class} "
        f"({predicted_confidence * 100:.1f}%) | "
        f"Display={class_display} | "
        f"YOLO={yolo_time:.3f}s | "
        f"Total={total_time:.3f}s",
        flush=True
    )

    print_detections(
        detections,
        image_width,
        image_height
    )

    return jsonify({
        "frame": encoded,
        "class": class_display,
        "confidence": final_confidence,
        "raw_class": predicted_class,
        "raw_confidence": predicted_confidence,
        "detections": detections,
        "processing_time": total_time
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("========================================")
    print("     Starting WasteVision Server")
    print("========================================")
    print(
        "Open: http://127.0.0.1:5000",
        flush=True
    )
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
