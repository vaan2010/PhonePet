import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const rotateGate = document.getElementById("rotateGate");
const video = document.getElementById("video");

const leftEyeEl = document.querySelector('.eye[data-side="left"]');
const rightEyeEl = document.querySelector('.eye[data-side="right"]');
const leftPupil = leftEyeEl.querySelector(".pupil");
const rightPupil = rightEyeEl.querySelector(".pupil");

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

const MAX_PUPIL_OFFSET = 28;
const BLINK_THRESHOLD = 0.18;
const SMOOTH = 0.35;
const MIRROR = true;

let faceLandmarker = null;
let running = false;
let starting = false;
let lastVideoTime = -1;
let isLandscape = false;

let smooth = {
  leftX: 0,
  leftY: 0,
  rightX: 0,
  rightY: 0,
  leftOpen: 1,
  rightOpen: 1,
};

function setStatus(text, show = true) {
  if (!text) {
    statusEl.hidden = true;
    statusEl.textContent = "";
    return;
  }
  statusEl.textContent = text;
  statusEl.hidden = !show;
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

function checkLandscape() {
  const landscapeQuery = window.matchMedia("(orientation: landscape)");
  if (typeof landscapeQuery.matches === "boolean") {
    return landscapeQuery.matches;
  }
  return window.innerWidth > window.innerHeight;
}

function updateOrientation() {
  isLandscape = checkLandscape();
  document.body.classList.toggle("is-landscape", isLandscape);
  document.body.classList.toggle("is-portrait", !isLandscape);
  rotateGate.hidden = isLandscape;

  // 直式時暫停畫面更新感，但相機可維持（避免重複要權限）
  if (!isLandscape) {
    startBtn.hidden = true;
  } else if (!running && !starting) {
    startBtn.hidden = false;
    startBtn.textContent = "點一下開始";
  }
}

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

  const flip = MIRROR ? -1 : 1;
  leftPupil.style.transform = `translate(${smooth.leftX * flip * MAX_PUPIL_OFFSET}%, ${smooth.leftY * MAX_PUPIL_OFFSET}%)`;
  rightPupil.style.transform = `translate(${smooth.rightX * flip * MAX_PUPIL_OFFSET}%, ${smooth.rightY * MAX_PUPIL_OFFSET}%)`;

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

async function createLandmarker() {
  setStatus("載入模型中…");
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
  if (!isLandscape || running || starting) return;

  starting = true;
  startBtn.disabled = true;
  startBtn.textContent = "啟動中…";

  try {
    if (!faceLandmarker) {
      await createLandmarker();
    }

    setStatus("請求相機權限…");
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
    running = true;
    startBtn.hidden = true;
    setStatus("");
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    starting = false;
    startBtn.disabled = false;
    startBtn.hidden = false;
    startBtn.textContent = "點一下開始";

    if (err?.name === "NotAllowedError") {
      setStatus("相機權限被拒絕");
    } else if (location.protocol !== "https:" && location.hostname !== "localhost") {
      setStatus("需要 HTTPS 才能開相機");
    } else {
      setStatus("無法開啟相機");
    }
  } finally {
    starting = false;
  }
}

function loop() {
  if (!running || !faceLandmarker) return;

  if (isLandscape && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = faceLandmarker.detectForVideo(video, performance.now());

    if (result.faceLandmarks?.length) {
      const landmarks = result.faceLandmarks[0];
      if (landmarks.length > 473) {
        updateCartoonEyes(eyeGaze(landmarks, LEFT), eyeGaze(landmarks, RIGHT));
      }
    }
  }

  requestAnimationFrame(loop);
}

function onOrientationMaybeChanged() {
  updateOrientation();
}

startBtn.addEventListener("click", startCamera);

window.addEventListener("resize", onOrientationMaybeChanged);
window.addEventListener("orientationchange", onOrientationMaybeChanged);

const landscapeQuery = window.matchMedia("(orientation: landscape)");
if (landscapeQuery.addEventListener) {
  landscapeQuery.addEventListener("change", onOrientationMaybeChanged);
} else if (landscapeQuery.addListener) {
  landscapeQuery.addListener(onOrientationMaybeChanged);
}

updateOrientation();

if (!navigator.mediaDevices?.getUserMedia) {
  setStatus("此瀏覽器不支援相機");
  startBtn.disabled = true;
}
