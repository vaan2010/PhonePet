import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const mirrorToggle = document.getElementById("mirrorToggle");
const ctx = overlay.getContext("2d");

const leftEyeEl = document.querySelector('.eye[data-side="left"]');
const rightEyeEl = document.querySelector('.eye[data-side="right"]');
const leftPupil = leftEyeEl.querySelector(".pupil");
const rightPupil = rightEyeEl.querySelector(".pupil");

/** MediaPipe face mesh indices */
const LEFT = {
  outer: 33,
  inner: 133,
  top: 159,
  bottom: 145,
  iris: 468,
};
const RIGHT = {
  outer: 263,
  inner: 362,
  top: 386,
  bottom: 374,
  iris: 473,
};

const MAX_PUPIL_OFFSET = 28; // % of eye box
const BLINK_THRESHOLD = 0.18;
const SMOOTH = 0.35;

let faceLandmarker = null;
let running = false;
let lastVideoTime = -1;
let smooth = {
  leftX: 0,
  leftY: 0,
  rightX: 0,
  rightY: 0,
  leftOpen: 1,
  rightOpen: 1,
};

function setStatus(text) {
  statusEl.textContent = text;
}

function applyMirrorClass() {
  const on = mirrorToggle.checked;
  video.classList.toggle("mirror", on);
  overlay.classList.toggle("mirror", on);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

/**
 * Gaze offset in roughly -1..1 relative to eye socket.
 * x+: look toward viewer's right (after mirror handling in render)
 * y+: look down
 */
function eyeGaze(landmarks, eye) {
  const outer = landmarks[eye.outer];
  const inner = landmarks[eye.inner];
  const top = landmarks[eye.top];
  const bottom = landmarks[eye.bottom];
  const iris = landmarks[eye.iris];

  const midX = (outer.x + inner.x) / 2;
  const midY = (top.y + bottom.y) / 2;
  const halfW = Math.max(dist(outer, inner) / 2, 1e-6);
  const halfH = Math.max(dist(top, bottom) / 2, 1e-6);

  // Face mesh x grows to the right of the image.
  // For selfie (often mirrored), "looking left" appears on the right side of the frame.
  const x = clamp((iris.x - midX) / halfW, -1.2, 1.2);
  const y = clamp((iris.y - midY) / halfH, -1.2, 1.2);
  const open = clamp(dist(top, bottom) / dist(outer, inner), 0, 1);

  return { x, y, open };
}

function updateCartoonEyes(left, right) {
  smooth.leftX = lerp(smooth.leftX, left.x, SMOOTH);
  smooth.leftY = lerp(smooth.leftY, left.y, SMOOTH);
  smooth.rightX = lerp(smooth.rightX, right.x, SMOOTH);
  smooth.rightY = lerp(smooth.rightY, right.y, SMOOTH);
  smooth.leftOpen = lerp(smooth.leftOpen, left.open, SMOOTH);
  smooth.rightOpen = lerp(smooth.rightOpen, right.open, SMOOTH);

  // Mirror mode: flip horizontal so cartoon eyes feel like a mirror.
  const flip = mirrorToggle.checked ? -1 : 1;

  const lx = smooth.leftX * flip * MAX_PUPIL_OFFSET;
  const ly = smooth.leftY * MAX_PUPIL_OFFSET;
  const rx = smooth.rightX * flip * MAX_PUPIL_OFFSET;
  const ry = smooth.rightY * MAX_PUPIL_OFFSET;

  leftPupil.style.transform = `translate(${lx}%, ${ly}%)`;
  rightPupil.style.transform = `translate(${rx}%, ${ry}%)`;

  setLid(leftEyeEl, smooth.leftOpen);
  setLid(rightEyeEl, smooth.rightOpen);
}

function setLid(eyeEl, openAmount) {
  const top = eyeEl.querySelector(".lid-top");
  const bottom = eyeEl.querySelector(".lid-bottom");
  const closed = openAmount < BLINK_THRESHOLD;
  const closeAmount = closed ? 1 : clamp(1 - openAmount / 0.28, 0, 1);
  const scale = 0.08 + closeAmount * 0.92;
  top.style.transform = `scaleY(${scale})`;
  bottom.style.transform = `scaleY(${scale})`;
}

function drawDebug(landmarks) {
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const drawingUtils = new DrawingUtils(ctx);
  drawingUtils.drawLandmarks(
    [landmarks[LEFT.iris], landmarks[RIGHT.iris]],
    { color: "#7eb6ff", radius: 3 }
  );

  for (const eye of [LEFT, RIGHT]) {
    const pts = [eye.outer, eye.inner, eye.top, eye.bottom].map((i) => landmarks[i]);
    ctx.beginPath();
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = 2;
    pts.forEach((p, idx) => {
      const x = p.x * overlay.width;
      const y = p.y * overlay.height;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  }
}

async function createLandmarker() {
  setStatus("載入臉部模型中…");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const modelAssetPath =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

  const options = {
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  };

  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath, delegate: "GPU" },
    });
  } catch (gpuErr) {
    console.warn("GPU delegate failed, fallback to CPU", gpuErr);
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath, delegate: "CPU" },
    });
  }
}

async function startCamera() {
  startBtn.disabled = true;
  try {
    if (!faceLandmarker) {
      await createLandmarker();
    }

    setStatus("請求相機權限中…");
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    });

    video.srcObject = stream;
    await video.play();
    applyMirrorClass();
    running = true;
    setStatus("追蹤中：動動眼睛試試看");
    startBtn.textContent = "執行中";
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    startBtn.disabled = false;
    if (err?.name === "NotAllowedError") {
      setStatus("相機權限被拒絕，請到 Safari 設定允許此網站");
    } else if (location.protocol !== "https:" && location.hostname !== "localhost") {
      setStatus("需要 HTTPS（可用 GitHub Pages）才能開相機");
    } else {
      setStatus("無法開啟相機：" + (err.message || err.name || "未知錯誤"));
    }
  }
}

function loop() {
  if (!running || !faceLandmarker) return;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = faceLandmarker.detectForVideo(video, performance.now());

    if (result.faceLandmarks?.length) {
      const landmarks = result.faceLandmarks[0];
      // Need iris landmarks (468+)
      if (landmarks.length > 473) {
        const left = eyeGaze(landmarks, LEFT);
        const right = eyeGaze(landmarks, RIGHT);
        updateCartoonEyes(left, right);
        drawDebug(landmarks);
      } else {
        setStatus("模型未輸出虹膜點，請重整再試");
      }
    }
  }

  requestAnimationFrame(loop);
}

startBtn.addEventListener("click", startCamera);
mirrorToggle.addEventListener("change", applyMirrorClass);
applyMirrorClass();

if (!navigator.mediaDevices?.getUserMedia) {
  setStatus("此瀏覽器不支援相機 API");
  startBtn.disabled = true;
}
