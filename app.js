import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const startPanel = document.getElementById("startPanel");
const stylePicker = document.getElementById("stylePicker");
const rotateGate = document.getElementById("rotateGate");
const eyesEl = document.getElementById("eyes");
const video = document.getElementById("video");
const motionCanvas = document.getElementById("motionCanvas");
const motionCtx = motionCanvas.getContext("2d", { willReadFrequently: true });

const STYLE_KEY = "desktop-pet-eye-style";
const STYLES = ["looi", "orb", "mono", "led"];

const leftEyeEl = document.querySelector('.eye[data-side="left"]');
const rightEyeEl = document.querySelector('.eye[data-side="right"]');
const leftPupil = leftEyeEl.querySelector(".pupil");
const rightPupil = rightEyeEl.querySelector(".pupil");

const LEFT_OUTER = 33;
const RIGHT_OUTER = 263;
const NOSE = 1;

const MAX_PUPIL_OFFSET = 30;
const BLINK_OPEN_THRESHOLD = 0.17;
const SMOOTH = 0.22;

const EMOTIONS = {
  idle: { blinkEvery: [2200, 4200], open: 0.92 },
  happy: { blinkEvery: [1800, 3200], open: 0.82 },
  confused: { blinkEvery: [2600, 4800], open: 0.88 },
  sleepy: { blinkEvery: [3500, 6500], open: 0.45 },
  startled: { blinkEvery: [99999, 99999], open: 1 },
  shy: { blinkEvery: [99999, 99999], open: 0.22 },
  curious: { blinkEvery: [1600, 2800], open: 0.96 },
  recoil: { blinkEvery: [2000, 3600], open: 0.78 },
};

let faceLandmarker = null;
let running = false;
let starting = false;
let lastVideoTime = -1;
let isLandscape = false;

let emotion = "idle";
let emotionUntil = 0;
let lockedEmotion = null;

let look = { x: 0, y: 0 };
let smoothLook = { x: 0, y: 0 };
let targetLook = { x: 0, y: 0 };
let pupilScale = 1;

let facePresent = false;
let baselineFaceSize = null;
let lastFaceSeenAt = 0;

let autoLook = { x: 0, y: 0, until: 0 };
let blinkUntil = 0;
let nextBlinkAt = 0;
let nextAutoAt = 0;
let yawnUntil = 0;
let interactionUntil = 0;
let lastUserEmotionAt = 0;

let prevMotionFrame = null;
let lastWaveAt = 0;
let prevUserBlink = false;

function now() {
  return performance.now();
}

function setStatus(text) {
  if (!text) {
    statusEl.hidden = true;
    statusEl.textContent = "";
    return;
  }
  statusEl.textContent = text;
  statusEl.hidden = false;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function rand(min, max) {
  return min + Math.random() * (max - min);
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function checkLandscape() {
  const q = window.matchMedia("(orientation: landscape)");
  if (typeof q.matches === "boolean") return q.matches;
  return window.innerWidth > window.innerHeight;
}

function applyStyle(style) {
  const next = STYLES.includes(style) ? style : "looi";
  document.body.dataset.style = next;
  localStorage.setItem(STYLE_KEY, next);
  for (const btn of stylePicker.querySelectorAll(".style-card")) {
    btn.setAttribute("aria-pressed", String(btn.dataset.style === next));
  }
}

function updateOrientation() {
  isLandscape = checkLandscape();
  document.body.classList.toggle("is-landscape", isLandscape);
  document.body.classList.toggle("is-portrait", !isLandscape);
  rotateGate.hidden = isLandscape;

  if (!isLandscape) {
    startPanel.hidden = true;
  } else if (!running && !starting) {
    startPanel.hidden = false;
  }
}

function setEmotion(next, durationMs = 0) {
  emotion = next;
  eyesEl.dataset.emotion = next;
  emotionUntil = durationMs > 0 ? now() + durationMs : 0;
  scheduleBlink();
}

function lockEmotion(next, durationMs) {
  lockedEmotion = next;
  setEmotion(next, durationMs);
  interactionUntil = now() + durationMs;
}

function scheduleBlink() {
  const conf = EMOTIONS[emotion] || EMOTIONS.idle;
  nextBlinkAt = now() + rand(conf.blinkEvery[0], conf.blinkEvery[1]);
}

function blendMap(categories) {
  const map = Object.create(null);
  if (!categories) return map;
  for (const c of categories) map[c.categoryName] = c.score;
  return map;
}

/**
 * 只觀察使用者臉的位置，讓機器人「看向你」。
 * 不同步使用者眼球／眨眼。
 */
function observeFace(landmarks) {
  const nose = landmarks[NOSE];
  // 相機未鏡像：畫面左 = 使用者右；機器人看向你時水平取反
  const faceX = clamp((0.5 - nose.x) * 2.0, -1.1, 1.1);
  const faceY = clamp((nose.y - 0.45) * 2.2, -0.9, 0.9);
  const faceSize = dist(landmarks[LEFT_OUTER], landmarks[RIGHT_OUTER]);
  return { faceX, faceY, faceSize };
}

/**
 * 從使用者表情推估狀態，驅動機器人相對反應（不是模仿眼睛）。
 */
function reactToUserExpression(blendshapes) {
  if (now() < interactionUntil) return;
  if (now() - lastUserEmotionAt < 900) return;

  const b = blendMap(blendshapes);
  const smile = ((b.mouthSmileLeft || 0) + (b.mouthSmileRight || 0)) / 2;
  const jaw = b.jawOpen || 0;
  const browUp = b.browInnerUp || 0;
  const surprise = ((b.eyeWideLeft || 0) + (b.eyeWideRight || 0)) / 2;
  const blink =
    ((b.eyeBlinkLeft || 0) + (b.eyeBlinkRight || 0)) / 2 > 0.55;

  // 使用者眨眼：機器人可好奇地回眨一下（反應，不是同步開合）
  if (blink && !prevUserBlink && facePresent) {
    blinkUntil = now() + 140;
    if (Math.random() > 0.55) {
      setEmotion("curious", 700);
      lastUserEmotionAt = now();
    }
  }
  prevUserBlink = blink;

  if (smile > 0.45) {
    lockEmotion("happy", 1400);
    autoLook = { x: targetLook.x * 0.3, y: -0.05, until: now() + 800 };
    lastUserEmotionAt = now();
    return;
  }

  if (jaw > 0.45) {
    // 使用者張嘴／打哈欠 → 機器人困了跟著哈欠
    startYawn();
    lastUserEmotionAt = now();
    return;
  }

  if (surprise > 0.4 || browUp > 0.55) {
    lockEmotion(surprise > 0.45 ? "startled" : "confused", 1100);
    autoLook = {
      x: rand(-0.25, 0.25),
      y: -0.12,
      until: now() + 700,
    };
    pupilScale = surprise > 0.45 ? 1.18 : 1;
    lastUserEmotionAt = now();
  }
}

function updateDistanceEmotion(faceSizeValue) {
  if (now() < interactionUntil) return;

  if (baselineFaceSize == null) {
    baselineFaceSize = faceSizeValue;
    return;
  }

  baselineFaceSize = lerp(baselineFaceSize, faceSizeValue, 0.035);
  const ratio = faceSizeValue / Math.max(baselineFaceSize, 1e-6);

  if (ratio > 1.45) {
    setEmotion("recoil", 900);
    pupilScale = lerp(pupilScale, 0.72, 0.2);
  } else if (ratio > 1.18) {
    setEmotion("curious", 800);
    pupilScale = lerp(pupilScale, 1.15, 0.2);
  } else if (ratio < 0.72) {
    if (emotion === "curious" || emotion === "recoil") setEmotion("idle");
    pupilScale = lerp(pupilScale, 0.95, 0.1);
  } else {
    pupilScale = lerp(pupilScale, 1, 0.08);
  }
}

function detectWave() {
  if (!video.videoWidth) return false;

  const w = 80;
  const h = 60;
  motionCanvas.width = w;
  motionCanvas.height = h;
  motionCtx.drawImage(video, 0, 0, w, h);
  const frame = motionCtx.getImageData(0, 0, w, h).data;

  if (!prevMotionFrame) {
    prevMotionFrame = new Uint8ClampedArray(frame);
    return false;
  }

  let motion = 0;
  for (let y = 0; y < h * 0.65; y += 2) {
    for (let x = 0; x < w; x += 2) {
      const i = (y * w + x) * 4;
      const diff =
        Math.abs(frame[i] - prevMotionFrame[i]) +
        Math.abs(frame[i + 1] - prevMotionFrame[i + 1]) +
        Math.abs(frame[i + 2] - prevMotionFrame[i + 2]);
      if (diff > 70) motion += 1;
    }
  }

  prevMotionFrame.set(frame);

  const threshold = facePresent ? 95 : 125;
  if (motion > threshold && now() - lastWaveAt > 1800) {
    lastWaveAt = now();
    return true;
  }
  return false;
}

function pickAutonomousAction() {
  if (now() < interactionUntil || now() < yawnUntil) return;
  if (now() < nextAutoAt) return;

  // 有人在面前時少一點亂晃，多一點觀察；沒人時更常自己動
  nextAutoAt = now() + (facePresent ? rand(4200, 8200) : rand(2500, 5500));
  const roll = Math.random();

  if (facePresent) {
    if (roll < 0.35) {
      // 小小張望一下再看回使用者
      autoLook = {
        x: rand(-0.7, 0.7),
        y: rand(-0.2, 0.15),
        until: now() + rand(500, 900),
      };
    } else if (roll < 0.5) {
      lockEmotion("curious", 900);
    } else if (roll < 0.62) {
      setEmotion("confused", 1200);
    } else if (roll < 0.72) {
      startYawn();
    } else {
      setEmotion("happy", 1000);
    }
    return;
  }

  if (roll < 0.3) {
    autoLook = {
      x: rand(-0.95, 0.95),
      y: rand(-0.25, 0.25),
      until: now() + rand(700, 1400),
    };
    setEmotion("idle");
  } else if (roll < 0.45) {
    autoLook = { x: 0, y: 0, until: now() + 800 };
    lockEmotion("curious", 800);
  } else if (roll < 0.6) {
    startYawn();
  } else if (roll < 0.78) {
    setEmotion("sleepy", 2400);
    autoLook = { x: rand(-0.2, 0.2), y: 0.4, until: now() + 1800 };
  } else {
    setEmotion("confused", 1500);
  }
}

function startYawn() {
  yawnUntil = now() + 1600;
  lockEmotion("sleepy", 1600);
  blinkUntil = yawnUntil;
}

function maybeBlink() {
  if (now() < blinkUntil) return;
  if (emotion === "startled" || emotion === "shy") return;
  if (now() >= nextBlinkAt) {
    blinkUntil = now() + (emotion === "sleepy" ? 220 : 130);
    scheduleBlink();
  }
}

function resolveEmotion() {
  if (lockedEmotion && now() < interactionUntil) {
    emotion = lockedEmotion;
    eyesEl.dataset.emotion = emotion;
    return;
  }
  lockedEmotion = null;

  if (emotionUntil && now() >= emotionUntil) {
    emotionUntil = 0;
    setEmotion(facePresent ? "idle" : "sleepy");
  }

  if (!facePresent && now() - lastFaceSeenAt > 5000 && now() >= interactionUntil) {
    if (emotion !== "sleepy" && emotion !== "confused") {
      setEmotion(Math.random() > 0.5 ? "sleepy" : "idle");
    }
  }
}

function renderEyes() {
  smoothLook.x = lerp(smoothLook.x, look.x, SMOOTH);
  smoothLook.y = lerp(smoothLook.y, look.y, SMOOTH);

  const conf = EMOTIONS[emotion] || EMOTIONS.idle;
  const blinking = now() < blinkUntil;
  const yawning = now() < yawnUntil;

  let open = conf.open;
  if (blinking || yawning) open = 0.02;
  if (emotion === "startled") open = 1;
  if (emotion === "shy") {
    open = Math.min(open, 0.2);
    smoothLook.y = Math.max(smoothLook.y, 0.45);
  }

  const px = smoothLook.x * MAX_PUPIL_OFFSET;
  const py = smoothLook.y * MAX_PUPIL_OFFSET;

  leftPupil.style.transform = `translate(${px}%, ${py}%) scale(${pupilScale})`;
  rightPupil.style.transform = `translate(${px}%, ${py}%) scale(${pupilScale})`;

  setLid(leftEyeEl, open, emotion);
  setLid(rightEyeEl, open, emotion);
}

function setLid(eyeEl, openAmount, currentEmotion) {
  const top = eyeEl.querySelector(".lid-top");
  const bottom = eyeEl.querySelector(".lid-bottom");
  const side = eyeEl.dataset.side;
  const closed = openAmount < BLINK_OPEN_THRESHOLD;
  const closeAmount = closed ? 1 : clamp(1 - openAmount / 0.3, 0, 1);
  const base =
    Number.parseFloat(getComputedStyle(eyesEl).getPropertyValue("--lid-base")) || 0.1;

  let topScale = clamp(base + closeAmount * (1 - base), 0, 1);
  let bottomScale = Math.max(base * 0.7, closeAmount * 0.5);
  let skew = 0;

  if (!closed) {
    if (currentEmotion === "happy") bottomScale = Math.max(bottomScale, 0.4);
    if (currentEmotion === "sleepy") {
      topScale = Math.max(topScale, 0.5);
      bottomScale = Math.max(bottomScale, 0.28);
    }
    if (currentEmotion === "shy") {
      topScale = Math.max(topScale, 0.62);
      bottomScale = Math.max(bottomScale, 0.35);
    }
    if (currentEmotion === "confused") {
      if (side === "left") {
        topScale = Math.max(topScale, 0.34);
        skew = -8;
      } else {
        topScale = Math.max(topScale, 0.12);
        skew = 6;
      }
    }
    if (currentEmotion === "startled") {
      topScale = Math.min(topScale, 0.04);
      bottomScale = Math.min(bottomScale, 0.04);
    }
  }

  top.style.transform = `scaleY(${topScale}) skewX(${skew}deg)`;
  bottom.style.transform = `scaleY(${bottomScale})`;
}

function onEyesTap(event) {
  if (!running || !isLandscape) return;
  event.preventDefault();
  lockEmotion("shy", 1400);
  autoLook = { x: rand(-0.15, 0.15), y: 0.55, until: now() + 1200 };
  blinkUntil = now() + 900;
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
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
  };

  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath, delegate: "GPU" },
    });
  } catch (err) {
    console.warn("GPU failed, using CPU", err);
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
  const prevLabel = startBtn.textContent;
  startBtn.textContent = "喚醒中…";

  try {
    if (!faceLandmarker) await createLandmarker();

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
    startPanel.hidden = true;
    setStatus("");
    setEmotion("happy", 1200);
    scheduleBlink();
    nextAutoAt = now() + 2500;
    lastFaceSeenAt = now();
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    startBtn.disabled = false;
    startBtn.textContent = prevLabel || "喚醒機器人";
    startPanel.hidden = false;
    if (err?.name === "NotAllowedError") setStatus("相機權限被拒絕");
    else if (location.protocol !== "https:" && location.hostname !== "localhost") {
      setStatus("需要 HTTPS 才能開相機");
    } else setStatus("無法開啟相機");
  } finally {
    starting = false;
  }
}

function loop() {
  if (!running || !faceLandmarker) return;

  const t = now();

  if (isLandscape && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = faceLandmarker.detectForVideo(video, t);

    if (result.faceLandmarks?.length) {
      facePresent = true;
      lastFaceSeenAt = t;

      const observed = observeFace(result.faceLandmarks[0]);
      targetLook.x = observed.faceX;
      targetLook.y = observed.faceY;

      // 預設看向使用者臉，不跟眼球
      if (t >= autoLook.until && t >= interactionUntil) {
        look.x = targetLook.x;
        look.y = targetLook.y;
      }

      updateDistanceEmotion(observed.faceSize);

      const shapes = result.faceBlendshapes?.[0]?.categories;
      if (shapes) reactToUserExpression(shapes);
    } else {
      facePresent = false;
    }

    if (detectWave()) {
      const happy = Math.random() > 0.4;
      lockEmotion(happy ? "happy" : "startled", happy ? 1200 : 1000);
      autoLook = {
        x: facePresent ? targetLook.x : rand(-0.2, 0.2),
        y: happy ? -0.05 : -0.15,
        until: now() + 900,
      };
      if (!happy) pupilScale = 1.2;
    }
  }

  if (t < autoLook.until) {
    look.x = lerp(look.x, autoLook.x, 0.2);
    look.y = lerp(look.y, autoLook.y, 0.2);
  } else if (!facePresent && t >= interactionUntil) {
    look.x = lerp(look.x, autoLook.x || 0, 0.05);
    look.y = lerp(look.y, autoLook.y || 0.08, 0.05);
  }

  pickAutonomousAction();
  maybeBlink();
  resolveEmotion();
  renderEyes();

  requestAnimationFrame(loop);
}

stylePicker.addEventListener("click", (event) => {
  const card = event.target.closest(".style-card");
  if (!card) return;
  applyStyle(card.dataset.style);
});

startBtn.addEventListener("click", startCamera);
eyesEl.addEventListener("pointerdown", onEyesTap);

applyStyle(localStorage.getItem(STYLE_KEY) || "looi");

window.addEventListener("resize", updateOrientation);
window.addEventListener("orientationchange", updateOrientation);

const landscapeQuery = window.matchMedia("(orientation: landscape)");
if (landscapeQuery.addEventListener) {
  landscapeQuery.addEventListener("change", updateOrientation);
} else if (landscapeQuery.addListener) {
  landscapeQuery.addListener(updateOrientation);
}

updateOrientation();
scheduleBlink();
setEmotion("idle");

if (!navigator.mediaDevices?.getUserMedia) {
  setStatus("此瀏覽器不支援相機");
  startBtn.disabled = true;
}
