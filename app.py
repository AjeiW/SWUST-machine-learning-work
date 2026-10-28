"""Flask 本地网页界面：浏览器负责摄像头和框选，PyTorch 负责识别。"""

import argparse
import base64
import threading
import webbrowser
from pathlib import Path
from threading import Timer

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.serving import make_server

from model import MNISTCNN
from preprocess import preprocess_digit
from utils import CHECKPOINT_DIR, get_device, load_model_weights


PROJECT_DIR = Path(__file__).resolve().parent
WEB_DIR = PROJECT_DIR / "web"
HOST = "127.0.0.1"
DEFAULT_PORT = 5000
MAX_IMAGE_BYTES = 16 * 1024 * 1024

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES

model = None
device = None
model_lock = threading.Lock()
http_server = None


def initialize_model():
    """加载项目训练出的 CNN 权重。"""
    global model, device
    device = get_device()
    model = MNISTCNN().to(device)
    load_model_weights(model, CHECKPOINT_DIR / "best_mnist_cnn.pth", device)
    model.eval()
    print(f"Device: {device}")


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify(ready=model is not None, device=str(device) if device else None)


@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify(error="Model is not loaded"), 503

    encoded = np.frombuffer(request.get_data(cache=False), dtype=np.uint8)
    if encoded.size == 0:
        return jsonify(error="No image data received"), 400
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(error="Could not decode image"), 400

    processed = preprocess_digit(image)
    if not processed.valid:
        return jsonify(error=processed.message), 422

    try:
        with model_lock, torch.inference_mode():
            logits = model(processed.tensor.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted = int(probabilities.argmax().item())
            confidence = float(probabilities[predicted].item())
    except Exception:
        app.logger.exception("Model inference failed")
        return jsonify(error="Prediction failed; check the server console"), 500

    ok, preview_png = cv2.imencode(".png", processed.digit)
    if not ok:
        return jsonify(error="Could not encode preview image"), 500
    preview = base64.b64encode(preview_png.tobytes()).decode("ascii")
    return jsonify(
        digit=predicted,
        confidence=confidence,
        preview=preview,
    )


@app.post("/api/shutdown")
def shutdown():
    """由本机网页退出按钮安全停止 Flask 服务。"""
    if http_server is not None:
        threading.Thread(target=http_server.shutdown, name="server-shutdown", daemon=True).start()
    return jsonify(message="Server is stopping")


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST CNN 本地网页识别界面")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="本地网页端口，默认 5000")
    parser.add_argument("--no-browser", action="store_true", help="启动服务但不自动打开浏览器")
    return parser.parse_args()


def main():
    global http_server
    args = parse_args()
    initialize_model()

    url = f"http://{HOST}:{args.port}/"
    http_server = make_server(HOST, args.port, app, threaded=True)
    print(f"Web app: {url}")
    print("Press Ctrl+C in this terminal to stop the server.")
    if not args.no_browser:
        Timer(0.8, webbrowser.open, args=(url,)).start()

    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        http_server.server_close()
        http_server = None


if __name__ == "__main__":
    main()
