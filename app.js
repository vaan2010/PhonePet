import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const startPanel = document.getElementById("startPanel");
const rotateGate = document.getElementById("rotateGate");
const faceEl = document.getElementById("face");
const fxEl = document.getElementById("fx");
const video = document.getElementById("video");
const motionCanvas = document.getElementById("motionCanvas");
const motionCtx = motionCanvas.getContext("2d", { willReadFrequently: true });

const leftEye = document.querySelector('.bot-eye[data-side="left"]');
const rightEye = document.querySelector('.bot-eye[data-side="right"]');

const LEFT_OUTER = 33;
const RIGHT_OUTER = 263;
const NOSE = 1;

const MAX_LOOK = 16; // svg units
const SMOOTH = 0.22;

/** 表情設定：形狀由 CSS data-emotion 控制，這裡管節奏與輔助 icon */
const EMOTIONS = {
  idle: { blinkEvery: [2200, 4200], icons: [] },
  happy: { blinkEvery: [1800, 3200], icons: ["spark"] },
  love: { blinkEvery: [2000, 3600], icons: ["heart", "heart"] },
  sad: { blinkEvery: [2800, 4800], icons: ["sweat"] },
  angry: { blinkEvery: [2400, 4000], icons: ["anger"] },
  confused: { blinkEvery: [2600, 4500], icons: ["question"] },
  thinking: { blinkEvery: [3000, 5000], icons: ["dots"] },
  sleepy: { blinkEvery: [3500, 6500], icons: ["zzz"] },
  bored: { blinkEvery: [3200, 5600], icons: ["dots"] },
  startled: { blinkEvery: [99999, 99999], icons: ["exclaim"] },
  shy: { blinkEvery: [99999, 99999], icons: ["sweat"] },
  curious: { blinkEvery: [1600, 2800], icons: ["spark"] },
  recoil: { blinkEvery: [2000, 3600], icons: ["sweat"] },
  dizzy: { blinkEvery: [99999, 99999], icons: ["dizzy"] },
  wink: { blinkEvery: [99999, 99999], icons: ["spark"] },
};

const ICON_GLYPH = {
  question: "?",
  exclaim: "!",
  zzz: "Zzz",
  heart: "♥",
  anger: "╬",
  sweat: "💧",
  spark: "✦",
  dots: "…",
  dizzy: "@",
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
let lastFxAt = 0;

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

function updateOrientation() {
  isLandscape = checkLandscape();
  document.body.classList.toggle("is-landscape", isLandscape);
  document.body.classList.toggle("is-portrait", !isLandscape);
  rotateGate.hidden = isLandscape;

  if (!isLandscape) startPanel.hidden = true;
  else if (!running && !starting) startPanel.hidden = false;
}

function spawnFx(kinds, linger = false) {
  if (!kinds?.length) return;
  if (now() - lastFxAt < 350) return;
  lastFxAt = now();

  const slots = [
    { left: "58%", top: "18%" },
    { left: "72%", top: "28%" },
    { left: "64%", top: "12%" },
    { left: "78%", top: "40%" },
    { left: "22%", top: "16%" },
  ];

  kinds.forEach((kind, index) => {
    const el = document.createElement("div");
    const cls = kind === "dizzy" ? "question" : kind;
    el.className = `fx-item ${cls}${linger ? " linger" : ""}`;
    el.textContent = ICON_GLYPH[kind] || kind;
    const slot = slots[index % slots.length];
    el.style.left = `calc(${slot.left} + ${rand(-3, 3)}%)`;
    el.style.top = `calc(${slot.top} + ${rand(-2, 2)}%)`;
    fxEl.appendChild(el);
    window.setTimeout(() => el.remove(), linger ? 1500 : 950);
  });
}

function setEmotion(next, durationMs = 0, withFx = true) {
  const prev = emotion;
  emotion = next;
  faceEl.dataset.emotion = next;
  emotionUntil = durationMs > 0 ? now() + durationMs : 0;
  scheduleBlink();

  if (withFx && next !== prev) {
    const icons = EMOTIONS[next]?.icons || [];
    if (icons.length) spawnFx(icons, next === "confused" || next === "thinking");
  }
}

function lockEmotion(next, durationMs, withFx = true) {
  lockedEmotion = next;
  setEmotion(next, durationMs, withFx);
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

function observeFace(landmarks) {
  const nose = landmarks[NOSE];
  const faceX = clamp((0.5 - nose.x) * 2.0, -1.1, 1.1);
  const faceY = clamp((nose.y - 0.45) * 2.2, -0.9, 0.9);
  const faceSize = dist(landmarks[LEFT_OUTER], landmarks[RIGHT_OUTER]);
  return { faceX, faceY, faceSize };
}

function reactToUserExpression(blendshapes) {
  if (now() < interactionUntil) return;
  if (now() - lastUserEmotionAt < 900) return;

  const b = blendMap(blendshapes);
  const smile = ((b.mouthSmileLeft || 0) + (b.mouthSmileRight || 0)) / 2;
  const frown = ((b.mouthFrownLeft || 0) + (b.mouthFrownRight || 0)) / 2;
  const jaw = b.jawOpen || 0;
  const browUp = b.browInnerUp || 0;
  const browDown = ((b.browDownLeft || 0) + (b.browDownRight || 0)) / 2;
  const surprise = ((b.eyeWideLeft || 0) + (b.eyeWideRight || 0)) / 2;
  const blink = ((b.eyeBlinkLeft || 0) + (b.eyeBlinkRight || 0)) / 2 > 0.55;

  if (blink && !prevUserBlink && facePresent) {
    blinkUntil = now() + 140;
    if (Math.random() > 0.6) {
      setEmotion("curious", 700);
      lastUserEmotionAt = now();
    }
  }
  prevUserBlink = blink;

  if (smile > 0.55) {
    lockEmotion(Math.random() > 0.65 ? "love" : "happy", 1500);
    lastUserEmotionAt = now();
    return;
  }
  if (smile > 0.4) {
    lockEmotion("happy", 1200);
    lastUserEmotionAt = now();
    return;
  }
  if (frown > 0.35 || browDown > 0.45) {
    lockEmotion(browDown > 0.55 ? "angry" : "sad", 1300);
    lastUserEmotionAt = now();
    return;
  }
  if (jaw > 0.45) {
    startYawn();
    lastUserEmotionAt = now();
    return;
  }
  if (surprise > 0.4 || browUp > 0.55) {
    if (surprise > 0.45) lockEmotion("startled", 1100);
    else lockEmotion("confused", 1400);
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
    pupilScale = lerp(pupilScale, 0.78, 0.2);
  } else if (ratio > 1.18) {
    setEmotion("curious", 800);
    pupilScale = lerp(pupilScale, 1.12, 0.2);
  } else if (ratio < 0.72) {
    if (emotion === "curious" || emotion === "recoil") setEmotion("bored", 1200);
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

  nextAutoAt = now() + (facePresent ? rand(3800, 7600) : rand(2400, 5200));
  const roll = Math.random();

  if (facePresent) {
    if (roll < 0.22) {
      autoLook = {
        x: rand(-0.8, 0.8),
        y: rand(-0.25, 0.2),
        until: now() + rand(500, 900),
      };
    } else if (roll < 0.36) {
      lockEmotion("curious", 900);
    } else if (roll < 0.48) {
      lockEmotion("confused", 1600);
    } else if (roll < 0.58) {
      lockEmotion("thinking", 1600);
      autoLook = { x: 0.35, y: -0.45, until: now() + 1200 };
    } else if (roll < 0.68) {
      startYawn();
    } else if (roll < 0.78) {
      lockEmotion("wink", 900);
    } else if (roll < 0.88) {
      lockEmotion("happy", 1100);
    } else {
      lockEmotion("love", 1200);
    }
    return;
  }

  if (roll < 0.28) {
    autoLook = {
      x: rand(-0.95, 0.95),
      y: rand(-0.25, 0.25),
      until: now() + rand(700, 1400),
    };
    setEmotion("idle");
  } else if (roll < 0.42) {
    lockEmotion("bored", 1800);
  } else if (roll < 0.58) {
    startYawn();
  } else if (roll < 0.72) {
    setEmotion("sleepy", 2400);
  } else if (roll < 0.85) {
    lockEmotion("confused", 1500);
  } else {
    lockEmotion("dizzy", 1200);
  }
}

function startYawn() {
  yawnUntil = now() + 1600;
  lockEmotion("sleepy", 1600);
  blinkUntil = yawnUntil;
  spawnFx(["zzz"], true);
}

function maybeBlink() {
  if (now() < blinkUntil) return;
  if (["startled", "shy", "wink", "dizzy", "love"].includes(emotion)) return;
  if (now() >= nextBlinkAt) {
    blinkUntil = now() + (emotion === "sleepy" ? 220 : 120);
    scheduleBlink();
  }
}

function resolveEmotion() {
  if (lockedEmotion && now() < interactionUntil) {
    emotion = lockedEmotion;
    faceEl.dataset.emotion = emotion;
    return;
  }
  lockedEmotion = null;

  if (emotionUntil && now() >= emotionUntil) {
    emotionUntil = 0;
    setEmotion(facePresent ? "idle" : "sleepy", 0, false);
  }

  if (!facePresent && now() - lastFaceSeenAt > 5000 && now() >= interactionUntil) {
    if (emotion !== "sleepy" && emotion !== "bored") {
      setEmotion(Math.random() > 0.5 ? "sleepy" : "bored", 0, false);
    }
  }
}

function renderEyes() {
  smoothLook.x = lerp(smoothLook.x, look.x, SMOOTH);
  smoothLook.y = lerp(smoothLook.y, look.y, SMOOTH);

  const blinking = now() < blinkUntil || now() < yawnUntil;
  faceEl.dataset.blink = blinking ? "1" : "0";

  let lx = smoothLook.x;
  let ly = smoothLook.y;
  if (emotion === "shy") ly = Math.max(ly, 0.45);
  if (emotion === "thinking") {
    lx = Math.max(lx, 0.25);
    ly = Math.min(ly, -0.35);
  }
  if (emotion === "sad") ly = Math.max(ly, 0.25);

  const tx = lx * MAX_LOOK;
  const ty = ly * MAX_LOOK;
  const scale = pupilScale;

  for (const eye of [leftEye, rightEye]) {
    for (const layer of eye.querySelectorAll(".pupil-layer")) {
      layer.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    }
  }
}

function onFaceTap(event) {
  if (!running || !isLandscape) return;
  event.preventDefault();
  const roll = Math.random();
  if (roll < 0.45) {
    lockEmotion("shy", 1400);
    autoLook = { x: rand(-0.15, 0.15), y: 0.55, until: now() + 1200 };
    blinkUntil = now() + 900;
  } else if (roll < 0.7) {
    lockEmotion("wink", 900);
  } else if (roll < 0.85) {
    lockEmotion("love", 1200);
  } else {
    lockEmotion("dizzy", 1000);
  }
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
    lockEmotion("happy", 1200);
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
      const happy = Math.random() > 0.45;
      lockEmotion(happy ? "happy" : "startled", happy ? 1200 : 1000);
      autoLook = {
        x: facePresent ? targetLook.x : rand(-0.2, 0.2),
        y: happy ? -0.05 : -0.15,
        until: now() + 900,
      };
      if (!happy) pupilScale = 1.15;
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

startBtn.addEventListener("click", startCamera);
faceEl.addEventListener("pointerdown", onFaceTap);

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
setEmotion("idle", 0, false);

if (!navigator.mediaDevices?.getUserMedia) {
  setStatus("此瀏覽器不支援相機");
  startBtn.disabled = true;
}
