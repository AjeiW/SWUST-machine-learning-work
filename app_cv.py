"""OpenCV 单窗口摄像头演示：拖动框选 ROI，按空格键识别。"""

import argparse
from functools import lru_cache
import os
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from model import MNISTCNN
from preprocess import preprocess_digit
from utils import CHECKPOINT_DIR, get_device, load_model_weights


LEFT_WIDTH = 640
PANEL_HEIGHT = 600
RIGHT_WIDTH = 460
WINDOW_TITLE = "MNIST Digit Recognition"
CAMERA_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST CNN 摄像头与图片识别")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="识别本地图片文件；指定后不打开摄像头",
    )
    return parser.parse_args()


def point_to_frame(x, y, frame_shape):
    height, width = frame_shape[:2]
    frame_x = int(np.clip(round(x * width / LEFT_WIDTH), 0, width - 1))
    frame_y = int(np.clip(round(y * height / PANEL_HEIGHT), 0, height - 1))
    return frame_x, frame_y


def image_display_rect(image_shape):
    """返回上传图片在左侧预览区中的显示矩形。"""
    height, width = image_shape[:2]
    scale = min(LEFT_WIDTH / width, PANEL_HEIGHT / height)
    display_width = max(1, int(width * scale))
    display_height = max(1, int(height * scale))
    left = (LEFT_WIDTH - display_width) // 2
    top = (PANEL_HEIGHT - display_height) // 2
    return left, top, display_width, display_height


def point_to_image(x, y, image_shape):
    height, width = image_shape[:2]
    left, top, display_width, display_height = image_display_rect(image_shape)
    image_x = int(np.clip(round((x - left) * width / display_width), 0, width - 1))
    image_y = int(np.clip(round((y - top) * height / display_height), 0, height - 1))
    return image_x, image_y


def image_roi_to_display(roi, image_shape):
    image_height, image_width = image_shape[:2]
    left, top, display_width, display_height = image_display_rect(image_shape)
    x1, y1, x2, y2 = roi
    return (
        left + int(x1 * display_width / image_width),
        top + int(y1 * display_height / image_height),
        left + int(x2 * display_width / image_width),
        top + int(y2 * display_height / image_height),
    )


def normalize_roi(roi, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = roi
    x1, x2 = sorted((int(np.clip(x1, 0, width)), int(np.clip(x2, 0, width))))
    y1, y2 = sorted((int(np.clip(y1, 0, height)), int(np.clip(y2, 0, height))))
    return x1, y1, x2, y2


def mouse_callback(event, x, y, flags, state):
    """处理窗口按钮和左侧画面的 ROI 拖动。"""
    if event == cv2.EVENT_LBUTTONDOWN and LEFT_WIDTH <= x < LEFT_WIDTH + RIGHT_WIDTH:
        panel_x = x - LEFT_WIDTH
        camera_menu = state.get("camera_menu")
        if camera_menu is not None:
            row_start = 210
            row_height = 34
            for row, camera_index in enumerate(camera_menu):
                row_y = row_start + row * row_height
                if 30 <= panel_x <= 430 and row_y <= y <= row_y + 28:
                    state["camera_selection"] = camera_index
                    return
            cancel_y = row_start + len(camera_menu) * row_height + 8
            if 30 <= panel_x <= 430 and cancel_y <= y <= cancel_y + 30:
                state["camera_selection"] = -1
            return
        if 250 <= panel_x <= 445 and 12 <= y <= 54:
            state["upload_requested"] = True
        elif (250 <= panel_x <= 445 and 62 <= y <= 104
              and not state.get("camera_scan_pending")
              and state.get("camera_menu") is None):
            state["camera_requested"] = True
        return

    image = state.get("image")
    if image is not None:
        left, top, display_width, display_height = image_display_rect(image.shape)
        right = left + display_width
        bottom = top + display_height
        if event == cv2.EVENT_LBUTTONDOWN:
            if left <= x < right and top <= y < bottom:
                state["dragging"] = True
                state["drag_start"] = point_to_image(x, y, image.shape)
                state["roi"] = (*state["drag_start"], *state["drag_start"])
        elif event == cv2.EVENT_MOUSEMOVE and state.get("dragging"):
            x = int(np.clip(x, left, right))
            y = int(np.clip(y, top, bottom))
            end = point_to_image(x, y, image.shape)
            state["roi"] = (*state["drag_start"], *end)
        elif event == cv2.EVENT_LBUTTONUP and state.get("dragging"):
            state["dragging"] = False
            x = int(np.clip(x, left, right))
            y = int(np.clip(y, top, bottom))
            end = point_to_image(x, y, image.shape)
            state["roi"] = (*state["drag_start"], *end)
            state["roi_changed"] = True
        return
    frame_shape = state.get("frame_shape")
    if frame_shape is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if not (0 <= x < LEFT_WIDTH and 0 <= y < PANEL_HEIGHT):
            return
        state["dragging"] = True
        state["drag_start"] = point_to_frame(x, y, frame_shape)
        state["roi"] = (*state["drag_start"], *state["drag_start"])
    elif event == cv2.EVENT_MOUSEMOVE and state.get("dragging"):
        x = int(np.clip(x, 0, LEFT_WIDTH))
        y = int(np.clip(y, 0, PANEL_HEIGHT))
        end = point_to_frame(x, y, frame_shape)
        state["roi"] = (*state["drag_start"], *end)
    elif event == cv2.EVENT_LBUTTONUP and state.get("dragging"):
        state["dragging"] = False
        x = int(np.clip(x, 0, LEFT_WIDTH))
        y = int(np.clip(y, 0, PANEL_HEIGHT))
        end = point_to_frame(x, y, frame_shape)
        state["roi"] = (*state["drag_start"], *end)


def default_roi(frame_shape):
    height, width = frame_shape[:2]
    side = max(40, int(min(height, width) * 0.42))
    side = min(side, width, height)
    center_x, center_y = width // 2, height // 2
    return normalize_roi(
        (center_x - side // 2, center_y - side // 2,
         center_x + side // 2, center_y + side // 2),
        frame_shape,
    )


@lru_cache(maxsize=16)
def load_ui_font(size):
    """优先使用 Windows 系统自带的微软雅黑。"""
    font_paths = (
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyhbd.ttc",
    )
    for font_path in font_paths:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_texts(image_bgr, items):
    """用微软雅黑在 BGR 图像上批量绘制文字，整帧只转换一次。"""
    if not items:
        return image_bgr
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    drawer = ImageDraw.Draw(pil_image)
    for text, position, size, color in items:
        drawer.text(
            position,
            str(text),
            font=load_ui_font(size),
            fill=(int(color[2]), int(color[1]), int(color[0])),
        )
    image_bgr[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    return image_bgr


def draw_panel(result_text, confidence, preview, history, status, footer=None,
               camera_menu=None, camera_index=None, text_items=None):
    panel = np.zeros((PANEL_HEIGHT, RIGHT_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(panel, (250, 12), (445, 54), (60, 105, 150), -1)
    cv2.rectangle(panel, (250, 62), (445, 104), (75, 85, 75), -1)

    if preview is not None:
        enlarged = cv2.resize(preview, (168, 168), interpolation=cv2.INTER_NEAREST)
        enlarged = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)
        panel[210:378, 24:192] = enlarged
        cv2.rectangle(panel, (23, 209), (192, 378), (100, 100, 100), 1)

    if camera_menu is not None:
        row_start = 210
        row_height = 34
        cancel_y = row_start + len(camera_menu) * row_height + 8
        bottom = min(PANEL_HEIGHT - 8, cancel_y + 38)
        cv2.rectangle(panel, (18, 170), (442, bottom), (25, 32, 42), -1)
        cv2.rectangle(panel, (18, 170), (442, bottom), (100, 180, 220), 1)
        for row, index in enumerate(camera_menu):
            row_y = row_start + row * row_height
            color = (75, 95, 75) if index == camera_index else (55, 65, 78)
            cv2.rectangle(panel, (30, row_y), (430, row_y + 28), color, -1)
        cv2.rectangle(panel, (30, cancel_y), (430, cancel_y + 30), (85, 65, 65), -1)

    items = [
        ("MNIST CNN", (24, 16), 25, (100, 220, 255)),
        ("Upload Image", (265, 17), 17, (255, 255, 255)),
        ("Select Camera", (263, 67), 16, (255, 255, 255)),
        (f"Prediction: {result_text}", (24, 68), 20, (255, 255, 255)),
        (f"Confidence: {confidence * 100:.1f}%" if confidence is not None else "Confidence: --",
         (24, 108), 16, (190, 220, 190)),
        (f"Status: {status}", (24, 145), 14, (180, 180, 180)),
    ]
    if camera_menu is None:
        items.extend([
            ("28x28 input", (24, 183), 15, (210, 210, 210)),
            ("Recent (newest first)", (24, 402), 16, (210, 210, 210)),
        ])
        for index, entry in enumerate(list(history)[:5]):
            items.append((f"{index + 1}. {entry}", (28, 435 + index * 27),
                          14, (220, 220, 220)))
    else:
        items.append(("Select camera (click a device)", (30, 178), 15,
                      (255, 255, 255)))
        for row, index in enumerate(camera_menu):
            row_y = 210 + row * 34
            label = f"Camera {index}" + (" (current)" if index == camera_index else "")
            items.append((label, (42, row_y + 4), 15, (245, 245, 245)))
        items.append(("Cancel  (ESC)", (42, cancel_y + 5), 15, (255, 230, 230)))
    footer = footer or "SPACE predict | C clear | Q quit"
    items.append((footer, (20, 570), 12, (145, 145, 145)))

    if text_items is None:
        return draw_texts(panel, items)
    text_items.extend(items)
    return panel


def fit_image_to_panel(image):
    """保持原图宽高比，缩放并居中到左侧显示区。"""
    left, top, resized_width, resized_height = image_display_rect(image.shape)
    resized = cv2.resize(image, (resized_width, resized_height))
    view = np.zeros((PANEL_HEIGHT, LEFT_WIDTH, 3), dtype=np.uint8)
    view[top : top + resized_height, left : left + resized_width] = resized
    return view


def cache_uploaded_image(state, image):
    state["image"] = image
    state["image_view"] = fit_image_to_panel(image)
    state["roi"] = None
    state["dragging"] = False


def read_image(image_path):
    """读取本地图片，并兼容 Windows 下的中文路径。"""
    try:
        encoded = np.fromfile(image_path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def choose_image_file():
    """弹出系统文件选择框，取消时返回空字符串。"""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        # 先刷新隐藏的父窗口，避免文件对话框被压在 OpenCV 窗口后面。
        root.update_idletasks()
        root.update()
        return filedialog.askopenfilename(
            parent=root,
            title="Select a digit image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()


def list_available_cameras(max_index=5):
    """探测编号 0 到 max_index-1 的摄像头，并及时释放探测句柄。"""
    available = []
    for index in range(max_index):
        candidate = cv2.VideoCapture(index, CAMERA_BACKEND)
        if candidate.isOpened():
            available.append(index)
        candidate.release()
    return available


def open_camera(index):
    camera = cv2.VideoCapture(index, CAMERA_BACKEND)
    if not camera.isOpened():
        camera.release()
        return None
    # 降低高分辨率采集和过期缓冲帧带来的额外负载。
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def start_camera_scan(state, mode, previous_index=None):
    """在后台探测设备，避免扫描摄像头时卡住 OpenCV 界面。"""
    state["camera_scan_pending"] = True
    state["camera_scan_result"] = None
    state["camera_scan_cancelled"] = False
    state["camera_scan_mode"] = mode
    state["camera_previous_index"] = previous_index

    def scan():
        try:
            result = (list_available_cameras(), None)
        except Exception as error:
            result = ([], type(error).__name__)
        state["camera_scan_result"] = result

    threading.Thread(target=scan, name="camera-scan", daemon=True).start()


def predict_image(image, model, device):
    """预处理并预测一张图，返回预测、置信度、预览图和状态。"""
    processed = preprocess_digit(image)
    if not processed.valid:
        return "--", None, None, processed.message

    with torch.inference_mode():
        logits = model(processed.tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted].item())
    return str(predicted), confidence, processed.digit, "Recognition complete"


def main():
    args = parse_args()
    device = get_device()
    model = MNISTCNN().to(device)
    load_model_weights(model, CHECKPOINT_DIR / "best_mnist_cnn.pth", device)
    model.eval()
    print(f"Device: {device}")

    camera_index = args.camera
    camera = None
    state = {
        "frame_shape": None,
        "roi": None,
        "drag_start": None,
        "dragging": False,
        "upload_requested": False,
        "camera_requested": False,
        "camera_menu": None,
        "camera_selection": None,
        "camera_previous_index": None,
        "camera_scan_pending": False,
        "camera_scan_result": None,
        "camera_scan_cancelled": False,
        "camera_scan_mode": None,
        "roi_changed": False,
        "image": None,
        "image_view": None,
    }
    result_text = "--"
    confidence = None
    preview = None
    status = "Draw a digit, then press SPACE"
    history = deque(maxlen=5)

    if args.image:
        image = read_image(args.image)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
        cache_uploaded_image(state, image)
        result_text, confidence, preview, status = predict_image(
            image, model, device
        )
        status = f"Image: {status}"
        if confidence is not None:
            history.appendleft(f"{result_text} ({confidence * 100:.1f}%)")
    else:
        camera = open_camera(camera_index)
        if camera is None:
            status = "Scanning for cameras..."

    try:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, LEFT_WIDTH + RIGHT_WIDTH, PANEL_HEIGHT)
        cv2.setMouseCallback(WINDOW_TITLE, mouse_callback, state)
        if not args.image and camera is None:
            start_camera_scan(state, "startup")

        while True:
            frame = None
            camera_texts = []
            if state["roi_changed"]:
                if state["image"] is not None:
                    status = "Image region selected; press SPACE to recognize"
                state["roi_changed"] = False
            if state["image"] is not None:
                frame = state["image"]
                camera_view = state["image_view"].copy()
                camera_texts.append(("Drag to select | SPACE: predict", (16, 12),
                                     16, (255, 255, 255)))
                if state["roi"] is not None:
                    roi = normalize_roi(state["roi"], frame.shape)
                    display_roi = image_roi_to_display(roi, frame.shape)
                    cv2.rectangle(camera_view, display_roi[:2], display_roi[2:],
                                  (0, 255, 0), 2)
            elif camera is not None:
                ok, frame = camera.read()
                if not ok or frame is None:
                    status = "Camera frame unavailable"
                    camera_view = np.zeros((PANEL_HEIGHT, LEFT_WIDTH, 3), dtype=np.uint8)
                    camera_texts.append(("Camera frame unavailable", (16, 14),
                                         17, (255, 255, 255)))
                else:
                    camera_view = cv2.resize(frame, (LEFT_WIDTH, PANEL_HEIGHT))
                    state["frame_shape"] = frame.shape
                    if state["roi"] is None:
                        state["roi"] = default_roi(frame.shape)
                    state["roi"] = normalize_roi(state["roi"], frame.shape)
                    x1, y1, x2, y2 = state["roi"]
                    display_x1 = int(x1 * LEFT_WIDTH / frame.shape[1])
                    display_y1 = int(y1 * PANEL_HEIGHT / frame.shape[0])
                    display_x2 = int(x2 * LEFT_WIDTH / frame.shape[1])
                    display_y2 = int(y2 * PANEL_HEIGHT / frame.shape[0])
                    cv2.rectangle(camera_view, (display_x1, display_y1),
                                  (display_x2, display_y2), (0, 255, 0), 2)
                    camera_texts.append(("Drag ROI | SPACE: predict | R: reset",
                                         (16, 12), 16, (255, 255, 255)))
            else:
                camera_view = np.zeros((PANEL_HEIGHT, LEFT_WIDTH, 3), dtype=np.uint8)
                camera_texts.append((
                    "Scanning cameras..." if state["camera_scan_pending"]
                    else "Select a camera using the button",
                    (16, 14), 17, (255, 255, 255),
                ))

            footer = "Click device | ESC cancel | Q quit" if state["camera_menu"] is not None else (
                "Scanning | ESC cancel | Q quit" if state["camera_scan_pending"] else None
            )
            panel_texts = []
            panel = draw_panel(
                result_text, confidence, preview, history, status, footer,
                state["camera_menu"], camera_index, panel_texts,
            )
            canvas = np.hstack((camera_view, panel))
            all_texts = list(camera_texts)
            all_texts.extend(
                (text, (x + LEFT_WIDTH, y), size, color)
                for text, (x, y), size, color in panel_texts
            )
            draw_texts(canvas, all_texts)
            cv2.imshow(WINDOW_TITLE, canvas)
            key = cv2.waitKey(5 if camera is not None else 30) & 0xFF

            # Q 始终退出程序；ESC 只取消设备扫描或选择，不承担程序退出功能。
            if key in (ord("q"), ord("Q")):
                break
            if key == 27:
                if state["camera_menu"] is not None:
                    previous_index = state["camera_previous_index"]
                    state["camera_menu"] = None
                    state["camera_selection"] = None
                    state["camera_previous_index"] = None
                    camera = open_camera(previous_index) if previous_index is not None else None
                    status = "Camera selection cancelled"
                elif state["camera_scan_pending"]:
                    state["camera_scan_cancelled"] = True
                    status = "Cancelling camera scan..."

            # 点击窗口右上角关闭按钮时也退出主循环。
            try:
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if state["upload_requested"]:
                state["upload_requested"] = False
                try:
                    image_path = choose_image_file()
                    if image_path:
                        image = read_image(image_path)
                        if image is None:
                            status = "Could not read selected image"
                        else:
                            if state["camera_scan_pending"]:
                                state["camera_scan_cancelled"] = True
                            cache_uploaded_image(state, image)
                            result_text, confidence, preview, status = predict_image(
                                image, model, device
                            )
                            status = f"Image: {status}"
                            if confidence is not None:
                                history.appendleft(
                                    f"{result_text} ({confidence * 100:.1f}%)"
                                )
                except Exception as error:
                    status = f"Image selection failed: {type(error).__name__}"
                continue

            if state["camera_requested"]:
                state["camera_requested"] = False
                previous_index = camera_index if camera is not None else None
                if camera is not None:
                    camera.release()
                    camera = None
                status = "Scanning camera devices..."
                start_camera_scan(state, "select", previous_index)
                continue

            if state["camera_selection"] is not None:
                selected_index = state["camera_selection"]
                previous_index = state["camera_previous_index"]
                state["camera_selection"] = None
                state["camera_menu"] = None
                state["camera_previous_index"] = None
                if selected_index == -1:
                    camera = open_camera(previous_index) if previous_index is not None else None
                    status = "Camera selection cancelled"
                else:
                    selected_camera = open_camera(selected_index)
                    if selected_camera is None:
                        camera = open_camera(previous_index) if previous_index is not None else None
                        status = f"Could not open camera {selected_index}"
                    else:
                        camera = selected_camera
                        camera_index = selected_index
                        state["image"] = None
                        state["image_view"] = None
                        state["roi"] = None
                        state["frame_shape"] = None
                        status = f"Camera {camera_index} selected"
                continue

            if state["camera_scan_result"] is not None:
                available, error_name = state["camera_scan_result"]
                state["camera_scan_result"] = None
                state["camera_scan_pending"] = False
                scan_mode = state["camera_scan_mode"]
                previous_index = state["camera_previous_index"]
                state["camera_scan_mode"] = None
                if state["camera_scan_cancelled"]:
                    state["camera_scan_cancelled"] = False
                    camera = open_camera(previous_index) if previous_index is not None else None
                    state["camera_previous_index"] = None
                    status = "Camera scan cancelled"
                elif error_name is not None:
                    camera = open_camera(previous_index) if previous_index is not None else None
                    state["camera_previous_index"] = None
                    status = f"Camera scan failed: {error_name}"
                elif scan_mode == "select":
                    if available:
                        state["camera_menu"] = available
                        status = "Click a camera in the right panel; ESC cancels"
                    else:
                        camera = open_camera(previous_index) if previous_index is not None else None
                        state["camera_previous_index"] = None
                        status = "No camera found"
                else:
                    camera = None
                    for candidate_index in available:
                        camera = open_camera(candidate_index)
                        if camera is not None:
                            camera_index = candidate_index
                            break
                    status = (
                        f"Camera {camera_index} opened"
                        if camera is not None
                        else "No camera found; use Upload Image or Select Camera"
                    )
                    state["camera_previous_index"] = None
                continue

            if key in (ord("r"), ord("R")):
                if state["image"] is not None:
                    state["roi"] = None
                    status = "Image selection reset; SPACE predicts full image"
                elif frame is not None:
                    state["roi"] = default_roi(frame.shape)
                    status = "ROI reset"
                else:
                    status = "Select Camera to return to camera view"
            elif key in (ord("c"), ord("C")):
                history.clear()
                result_text, confidence, preview = "--", None, None
                status = "History cleared"
            elif key == ord(" "):
                if state["image"] is not None:
                    image_to_predict = state["image"]
                    status_prefix = "Image"
                    if state["roi"] is not None:
                        x1, y1, x2, y2 = normalize_roi(state["roi"], frame.shape)
                        if x2 - x1 < 8 or y2 - y1 < 8:
                            status = "Selected region is too small; drag a larger box"
                            continue
                        image_to_predict = frame[y1:y2, x1:x2]
                        status_prefix = "Image region"
                    result_text, confidence, preview, status = predict_image(
                        image_to_predict, model, device
                    )
                    status = f"{status_prefix}: {status}"
                    if confidence is not None:
                        history.appendleft(f"{result_text} ({confidence * 100:.1f}%)")
                    continue
                if frame is None:
                    status = "Camera frame unavailable"
                    continue
                x1, y1, x2, y2 = normalize_roi(state["roi"], frame.shape)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    status = "ROI is too small; drag a larger box"
                    continue
                result_text, confidence, preview, status = predict_image(
                    frame[y1:y2, x1:x2], model, device
                )
                if confidence is not None:
                    history.appendleft(f"{result_text} ({confidence * 100:.1f}%)")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
