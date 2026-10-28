(() => {
  const video = document.getElementById("cameraVideo");
  const canvas = document.getElementById("previewCanvas");
  const canvasContext = canvas.getContext("2d");
  const sourceCanvas = document.createElement("canvas");
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: false });

  const emptyState = document.getElementById("emptyState");
  const cameraSelect = document.getElementById("cameraSelect");
  const startCameraButton = document.getElementById("startCameraButton");
  const stopCameraButton = document.getElementById("stopCameraButton");
  const refreshCameraButton = document.getElementById("refreshCameraButton");
  const imageInput = document.getElementById("imageInput");
  const recognizeButton = document.getElementById("recognizeButton");
  const resetRegionButton = document.getElementById("resetRegionButton");
  const exitButton = document.getElementById("exitButton");
  const clearHistoryButton = document.getElementById("clearHistoryButton");
  const predictionDigit = document.getElementById("predictionDigit");
  const confidenceValue = document.getElementById("confidenceValue");
  const processedImage = document.getElementById("processedImage");
  const processedPlaceholder = document.getElementById("processedPlaceholder");
  const statusMessage = document.getElementById("statusMessage");
  const sourceBadge = document.getElementById("sourceBadge");
  const dimensionsLabel = document.getElementById("imageDimensions");
  const historyList = document.getElementById("historyList");

  let stream = null;
  let mode = null;
  let imageSource = null;
  let imageObjectUrl = null;
  let roi = null;
  let dragStart = null;
  let dragging = false;
  let recognizing = false;
  let exiting = false;
  let cameraFrameRequest = 0;
  let lastCameraDraw = 0;
  let history = [];

  function setStatus(message, kind = "") {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${kind}`.trim();
  }

  function setCanvasSize(width, height) {
    const scale = Math.min(1, 1280 / Math.max(width, height));
    const targetWidth = Math.max(1, Math.round(width * scale));
    const targetHeight = Math.max(1, Math.round(height * scale));
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      sourceCanvas.width = targetWidth;
      sourceCanvas.height = targetHeight;
    }
    dimensionsLabel.textContent = `${targetWidth} × ${targetHeight}`;
  }

  function normalizeSelection() {
    if (!roi) return null;
    const x1 = Math.min(roi.x1, roi.x2);
    const y1 = Math.min(roi.y1, roi.y2);
    const x2 = Math.max(roi.x1, roi.x2);
    const y2 = Math.max(roi.y1, roi.y2);
    return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
  }

  function drawOverlay() {
    canvasContext.clearRect(0, 0, canvas.width, canvas.height);
    canvasContext.drawImage(sourceCanvas, 0, 0);
    const selection = normalizeSelection();
    if (selection && (selection.width > 0 || selection.height > 0)) {
      canvasContext.save();
      canvasContext.fillStyle = "rgba(16, 185, 129, 0.12)";
      canvasContext.strokeStyle = "#22d3a6";
      canvasContext.lineWidth = Math.max(2, canvas.width / 700);
      canvasContext.fillRect(selection.x, selection.y, selection.width, selection.height);
      canvasContext.strokeRect(selection.x, selection.y, selection.width, selection.height);
      canvasContext.restore();
    }
  }

  function updateControls() {
    const hasInput = mode !== null;
    recognizeButton.disabled = !hasInput || recognizing;
    resetRegionButton.disabled = !hasInput || roi === null;
    stopCameraButton.disabled = stream === null;
    startCameraButton.disabled = stream !== null;
    cameraSelect.disabled = cameraSelect.options.length <= 1;
  }

  function showSource(label) {
    emptyState.hidden = true;
    canvas.hidden = false;
    sourceBadge.textContent = label;
    sourceBadge.classList.add("active");
    updateControls();
  }

  function showEmpty(message = "打开摄像头或上传一张图片") {
    canvas.hidden = true;
    emptyState.hidden = false;
    emptyState.querySelector("strong").textContent = message;
    sourceBadge.textContent = "未选择输入";
    sourceBadge.classList.remove("active");
    dimensionsLabel.textContent = "—";
    updateControls();
  }

  async function refreshCameraDevices(preferredId = "") {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setStatus("当前浏览器不支持摄像头设备枚举", "error");
      return;
    }
    try {
      const devices = (await navigator.mediaDevices.enumerateDevices())
        .filter((item) => item.kind === "videoinput");
      const previousId = preferredId || cameraSelect.value;
      cameraSelect.replaceChildren();
      if (devices.length === 0) {
        cameraSelect.add(new Option("未发现摄像头设备", ""));
        cameraSelect.disabled = true;
        return;
      }
      devices.forEach((item, index) => {
        const label = item.label || `摄像头 ${index + 1}`;
        cameraSelect.add(new Option(label, item.deviceId));
      });
      if (previousId && devices.some((item) => item.deviceId === previousId)) {
        cameraSelect.value = previousId;
      }
      cameraSelect.disabled = false;
    } catch (error) {
      setStatus(`读取摄像头列表失败：${error.message}`, "error");
    }
  }

  function releaseCamera() {
    if (cameraFrameRequest) {
      cancelAnimationFrame(cameraFrameRequest);
      cameraFrameRequest = 0;
    }
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    video.srcObject = null;
    updateControls();
  }

  async function startCamera(deviceId = cameraSelect.value) {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("浏览器不支持摄像头访问；请使用 localhost 并检查权限", "error");
      return;
    }
    releaseCamera();
    mode = null;
    roi = null;
    updateControls();
    try {
      const cameraOptions = {
        audio: false,
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, max: 30 },
        },
      };
      if (deviceId) cameraOptions.video.deviceId = { exact: deviceId };
      stream = await navigator.mediaDevices.getUserMedia(cameraOptions);
      video.srcObject = stream;
      await video.play();
      mode = "camera";
      imageSource = null;
      showSource("摄像头画面");
      setStatus("摄像头已打开；拖动鼠标框选数字", "success");
      const activeDeviceId = stream.getVideoTracks()[0]?.getSettings().deviceId || deviceId;
      await refreshCameraDevices(activeDeviceId);
      renderCameraFrame();
    } catch (error) {
      releaseCamera();
      mode = null;
      showEmpty("摄像头无法打开，请检查权限或选择其他设备");
      setStatus(`摄像头打开失败：${error.message}`, "error");
      await refreshCameraDevices();
    }
  }

  function renderCameraFrame() {
    if (mode !== "camera" || !stream) return;
    const now = performance.now();
    if (now - lastCameraDraw < 1000 / 30 || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      cameraFrameRequest = requestAnimationFrame(renderCameraFrame);
      return;
    }
    lastCameraDraw = now;
    setCanvasSize(video.videoWidth, video.videoHeight);
    sourceContext.drawImage(video, 0, 0, sourceCanvas.width, sourceCanvas.height);
    drawOverlay();
    cameraFrameRequest = requestAnimationFrame(renderCameraFrame);
  }

  async function loadImageFile(file) {
    if (!file) return;
    releaseCamera();
    mode = null;
    roi = null;
    updateControls();
    try {
      if (imageObjectUrl) URL.revokeObjectURL(imageObjectUrl);
      imageObjectUrl = URL.createObjectURL(file);
      const loadedImage = new Image();
      loadedImage.src = imageObjectUrl;
      await loadedImage.decode();
      imageSource = loadedImage;
      mode = "image";
      setCanvasSize(loadedImage.naturalWidth, loadedImage.naturalHeight);
      sourceContext.drawImage(loadedImage, 0, 0, sourceCanvas.width, sourceCanvas.height);
      drawOverlay();
      showSource(`图片：${file.name}`);
      setStatus("图片已载入；拖动鼠标框选数字后点击识别", "success");
    } catch (error) {
      mode = null;
      imageSource = null;
      showEmpty("图片载入失败，请选择 PNG、JPG 或 BMP 图片");
      setStatus(`图片载入失败：${error.message}`, "error");
    }
  }

  function pointerPosition(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * canvas.width / rect.width)),
      y: Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * canvas.height / rect.height)),
    };
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (mode === null) return;
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    dragStart = pointerPosition(event);
    roi = { x1: dragStart.x, y1: dragStart.y, x2: dragStart.x, y2: dragStart.y };
    dragging = true;
    drawOverlay();
    updateControls();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const point = pointerPosition(event);
    roi = { x1: dragStart.x, y1: dragStart.y, x2: point.x, y2: point.y };
    drawOverlay();
  });

  function finishSelection(event) {
    if (!dragging) return;
    const point = pointerPosition(event);
    roi = { x1: dragStart.x, y1: dragStart.y, x2: point.x, y2: point.y };
    dragging = false;
    const selection = normalizeSelection();
    if (!selection || selection.width < 8 || selection.height < 8) {
      roi = null;
      setStatus("框选区域太小，请重新拖动选择", "error");
    } else {
      setStatus(`已框选 ${Math.round(selection.width)} × ${Math.round(selection.height)} 区域，按空格或点击识别`, "success");
    }
    drawOverlay();
    updateControls();
  }

  canvas.addEventListener("pointerup", finishSelection);
  canvas.addEventListener("pointercancel", finishSelection);

  async function recognize() {
    if (mode === null || recognizing) return;
    const selection = normalizeSelection();
    const x = selection ? Math.round(selection.x) : 0;
    const y = selection ? Math.round(selection.y) : 0;
    const width = selection ? Math.round(selection.width) : sourceCanvas.width;
    const height = selection ? Math.round(selection.height) : sourceCanvas.height;
    if (width < 8 || height < 8) {
      setStatus("识别区域太小，请选择更大的区域", "error");
      return;
    }

    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = width;
    cropCanvas.height = height;
    cropCanvas.getContext("2d").drawImage(
      sourceCanvas, x, y, width, height, 0, 0, width, height,
    );
    const blob = await new Promise((resolve) => cropCanvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) {
      setStatus("无法生成待识别图像", "error");
      return;
    }

    recognizing = true;
    updateControls();
    setStatus("正在识别…");
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: blob,
      });
      let result;
      try {
        result = await response.json();
      } catch {
        throw new Error(`服务返回异常（HTTP ${response.status}）`);
      }
      if (!response.ok) throw new Error(result.error || `识别失败（HTTP ${response.status}）`);

      predictionDigit.textContent = result.digit;
      confidenceValue.textContent = `${(result.confidence * 100).toFixed(2)}%`;
      processedImage.src = `data:image/png;base64,${result.preview}`;
      processedImage.hidden = false;
      processedPlaceholder.hidden = true;
      addHistory(result.digit, result.confidence);
      setStatus(selection ? "选区识别完成" : "整张图像识别完成", "success");
    } catch (error) {
      setStatus(error.message || "识别请求失败，请确认本机服务仍在运行", "error");
    } finally {
      recognizing = false;
      updateControls();
    }
  }

  function renderHistory() {
    historyList.replaceChildren();
    if (history.length === 0) {
      const empty = document.createElement("li");
      empty.className = "history-empty";
      empty.textContent = "还没有识别记录";
      historyList.append(empty);
      return;
    }
    history.forEach((entry, index) => {
      const row = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = `${index + 1}. ${entry.digit}`;
      const confidence = document.createElement("span");
      confidence.className = "history-confidence";
      confidence.textContent = `${(entry.confidence * 100).toFixed(1)}%`;
      row.append(label, confidence);
      historyList.append(row);
    });
  }

  function addHistory(digit, confidence) {
    history.unshift({ digit, confidence });
    history = history.slice(0, 5);
    renderHistory();
  }

  function resetRegion() {
    if (roi === null) return;
    roi = null;
    drawOverlay();
    updateControls();
    setStatus("已清除框选；下一次识别将使用整张图像");
  }

  async function exitApplication() {
    if (exiting) return;
    exiting = true;
    releaseCamera();
    setStatus("正在关闭本机服务…");
    exitButton.disabled = true;
    try {
      await fetch("/api/shutdown", { method: "POST" });
    } catch {
      // 服务停止后连接可能先断开，页面仍提示用户关闭标签页。
    }
    setStatus("服务已停止。可以关闭此浏览器标签页和终端。", "success");
  }

  startCameraButton.addEventListener("click", () => startCamera());
  stopCameraButton.addEventListener("click", () => {
    releaseCamera();
    mode = null;
    roi = null;
    showEmpty("摄像头已关闭；可重新打开或上传图片");
    setStatus("摄像头已关闭");
  });
  refreshCameraButton.addEventListener("click", () => refreshCameraDevices());
  cameraSelect.addEventListener("change", () => {
    if (cameraSelect.value) startCamera(cameraSelect.value);
  });
  imageInput.addEventListener("change", (event) => {
    loadImageFile(event.target.files?.[0]);
    event.target.value = "";
  });
  recognizeButton.addEventListener("click", recognize);
  resetRegionButton.addEventListener("click", resetRegion);
  clearHistoryButton.addEventListener("click", () => {
    history = [];
    renderHistory();
    setStatus("识别记录已清空");
  });
  exitButton.addEventListener("click", exitApplication);

  window.addEventListener("keydown", (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
    if (event.key.toLowerCase() === "q") {
      event.preventDefault();
      exitApplication();
    } else if (event.key === "Escape") {
      event.preventDefault();
      resetRegion();
    } else if (event.code === "Space") {
      event.preventDefault();
      recognize();
    } else if (event.key.toLowerCase() === "r") {
      resetRegion();
    } else if (event.key.toLowerCase() === "c") {
      history = [];
      renderHistory();
      setStatus("识别记录已清空");
    }
  });

  refreshCameraDevices();
  updateControls();
})();
