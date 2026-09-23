/*  FaceVision — app.js
 *  Real-time face detection, landmark detection, expression recognition
 *  and AR filters via face-api.js (TensorFlow.js backend)
 */

// ────────────────────────────────────────────────────────────────
//  CONFIG
// ────────────────────────────────────────────────────────────────
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model';

const EXPRESSION_MAP = {
  happy:     { label: 'Feliz',    emoji: '😄', color: '#34d399' },
  sad:       { label: 'Triste',   emoji: '😢', color: '#60a5fa' },
  angry:     { label: 'Raiva',    emoji: '😠', color: '#f87171' },
  surprised: { label: 'Surpreso', emoji: '😲', color: '#fbbf24' },
  fearful:   { label: 'Medo',     emoji: '😨', color: '#c084fc' },
  disgusted: { label: 'Nojo',     emoji: '🤢', color: '#86efac' },
  neutral:   { label: 'Neutro',   emoji: '😐', color: '#94a3b8' },
};

// ────────────────────────────────────────────────────────────────
//  DOM REFERENCES
// ────────────────────────────────────────────────────────────────
const video        = document.getElementById('video');
const overlayCanvas = document.getElementById('overlay-canvas');
const filterCanvas = document.getElementById('filter-canvas');
const overlayCtx   = overlayCanvas.getContext('2d');
const filterCtx    = filterCanvas.getContext('2d');

const loadingOverlay = document.getElementById('loading-overlay');
const startOverlay   = document.getElementById('start-overlay');
const startBtn       = document.getElementById('start-btn');
const toggleBtn      = document.getElementById('toggle-btn');
const mirrorBtn      = document.getElementById('mirror-btn');
const snapshotBtn    = document.getElementById('snapshot-btn');

const liveBadge    = document.getElementById('live-badge');
const modelBadge   = document.getElementById('model-badge');
const expressionHud = document.getElementById('expression-hud');
const faceCountNum  = document.getElementById('face-count-num');

const dominantEmoji = document.getElementById('dominant-emoji');
const dominantLabel = document.querySelector('.dominant-label');
const dominantSub   = document.getElementById('dominant-sub');

const toast           = document.getElementById('toast');
const snapshotModal   = document.getElementById('snapshot-modal');
const snapshotImg     = document.getElementById('snapshot-img');
const snapshotDl      = document.getElementById('snapshot-download');
const snapshotClose   = document.getElementById('snapshot-close');

const statFps     = document.getElementById('stat-fps');
const statFaces   = document.getElementById('stat-faces');
const statConf    = document.getElementById('stat-conf');
const statLatency = document.getElementById('stat-latency');

// ────────────────────────────────────────────────────────────────
//  STATE
// ────────────────────────────────────────────────────────────────
let stream       = null;
let detecting    = false;
let mirrored     = true;
let activeFilter = 'none';
let animFrameId  = null;
let fpsCounter   = 0;
let fpsTimer     = Date.now();
let lastDetections = [];

// ────────────────────────────────────────────────────────────────
//  LOAD MODELS
// ────────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
    ]);
    loadingOverlay.style.display = 'none';
    startOverlay.style.display   = 'flex';
    modelBadge.textContent = 'IA Pronta ✓';
    showToast('✅ Modelos de IA carregados com sucesso!');
  } catch (err) {
    console.error('Model load error:', err);
    // Fallback: try jsdelivr CDN for vladmandic models
    try {
      const ALT_URL = 'https://vladmandic.github.io/face-api/model';
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(ALT_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(ALT_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(ALT_URL),
      ]);
      loadingOverlay.style.display = 'none';
      startOverlay.style.display   = 'flex';
      modelBadge.textContent = 'IA Pronta ✓';
      showToast('✅ Modelos de IA carregados!');
    } catch (err2) {
      console.error('Alt model load error:', err2);
      loadingOverlay.querySelector('.loader-text').textContent = 'Erro ao carregar modelos';
      loadingOverlay.querySelector('.loader-sub').textContent  = 'Verifique sua conexão com a internet e recarregue a página.';
      modelBadge.textContent = 'Erro na IA';
    }
  }
}

// ────────────────────────────────────────────────────────────────
//  CAMERA
// ────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise(res => video.onloadedmetadata = res);
    video.play();

    // Size canvases to match video
    resizeCanvases();

    // Mirror by default
    applyMirror();

    // Update UI
    toggleBtn.textContent = '';
    toggleBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
        <rect x="2" y="2" width="20" height="20" rx="3"/>
        <line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/>
      </svg>
      Parar
    `;
    toggleBtn.classList.add('btn-stop');
    toggleBtn.disabled = false;
    mirrorBtn.disabled = false;
    snapshotBtn.disabled = false;

    liveBadge.classList.add('active');
    expressionHud.style.display = 'flex';
    document.querySelector('.stage-card').classList.add('detecting');

    startDetectionLoop();
    showToast('📷 Câmera ligada! Detecção em andamento...');
  } catch (err) {
    console.error('Camera error:', err);
    showToast('❌ Erro ao acessar câmera. Verifique as permissões.');
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  detecting = false;
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }

  video.srcObject = null;
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  filterCtx.clearRect(0, 0, filterCanvas.width, filterCanvas.height);

  toggleBtn.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
      <circle cx="12" cy="12" r="3"/>
      <path d="M19.07 4.93a10 10 0 0 0-14.14 0M4.93 19.07a10 10 0 0 0 14.14 0"/>
    </svg>
    Iniciar
  `;
  toggleBtn.classList.remove('btn-stop');
  mirrorBtn.disabled = true;
  snapshotBtn.disabled = true;
  liveBadge.classList.remove('active');
  expressionHud.style.display = 'none';
  document.querySelector('.stage-card').classList.remove('detecting');

  resetExpressionUI();
  showToast('⏹ Câmera desligada');
}

function resizeCanvases() {
  overlayCanvas.width  = video.videoWidth;
  overlayCanvas.height = video.videoHeight;
  filterCanvas.width   = video.videoWidth;
  filterCanvas.height  = video.videoHeight;
}

function applyMirror() {
  const transform = mirrored ? 'scaleX(-1)' : 'none';
  video.style.transform          = transform;
  overlayCanvas.style.transform  = transform;
  filterCanvas.style.transform   = transform;
}

// ────────────────────────────────────────────────────────────────
//  DETECTION LOOP
// ────────────────────────────────────────────────────────────────
async function startDetectionLoop() {
  detecting = true;

  const options = new faceapi.TinyFaceDetectorOptions({
    inputSize: 416,
    scoreThreshold: 0.5,
  });

  async function loop() {
    if (!detecting) return;

    const t0 = performance.now();

    try {
      const detections = await faceapi
        .detectAllFaces(video, options)
        .withFaceLandmarks()
        .withFaceExpressions();

      lastDetections = detections;

      const t1 = performance.now();
      const latency = Math.round(t1 - t0);

      // Resize to canvas coords
      const dims = { width: overlayCanvas.width, height: overlayCanvas.height };
      const resized = faceapi.resizeResults(detections, dims);

      // Clear canvases
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      filterCtx.clearRect(0, 0, filterCanvas.width, filterCanvas.height);

      // Draw filter first
      if (activeFilter !== 'none') {
        drawFilter(filterCtx, resized, activeFilter);
      }

      // Draw detection overlay
      drawDetectionOverlay(overlayCtx, resized);

      // Update UI
      updateExpressionUI(detections);
      updateStats(detections, latency);

      faceCountNum.textContent = detections.length;
    } catch (e) {
      // silently continue
    }

    // FPS counter
    fpsCounter++;
    const now = Date.now();
    if (now - fpsTimer >= 1000) {
      statFps.textContent = fpsCounter;
      fpsCounter = 0;
      fpsTimer = now;
    }

    animFrameId = requestAnimationFrame(loop);
  }

  loop();
}

// ────────────────────────────────────────────────────────────────
//  DETECTION OVERLAY
// ────────────────────────────────────────────────────────────────
function drawDetectionOverlay(ctx, detections) {
  detections.forEach((det, i) => {
    const { box } = det.detection;
    const expr = det.expressions;
    const dominant = getDominantExpression(expr);
    const info = EXPRESSION_MAP[dominant.key] || EXPRESSION_MAP.neutral;

    // Box
    ctx.save();
    ctx.strokeStyle = info.color;
    ctx.lineWidth = 2.5;
    ctx.shadowBlur = 16;
    ctx.shadowColor = info.color;
    ctx.beginPath();
    ctx.roundRect(box.x, box.y, box.width, box.height, 8);
    ctx.stroke();
    ctx.restore();

    // Corner accents
    drawCornerAccents(ctx, box, info.color);

    // Label above box
    const labelY = box.y - 10;
    const text = `${info.emoji} ${info.label} ${Math.round(dominant.value * 100)}%`;
    ctx.save();
    ctx.font = 'bold 14px Outfit, sans-serif';
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = 'rgba(13, 13, 31, 0.8)';
    ctx.beginPath();
    ctx.roundRect(box.x, labelY - 20, tw + 16, 24, 6);
    ctx.fill();
    ctx.fillStyle = info.color;
    ctx.shadowBlur = 8;
    ctx.shadowColor = info.color;
    ctx.fillText(text, box.x + 8, labelY - 3);
    ctx.restore();

    // Confidence indicator
    const conf = det.detection.score;
    ctx.save();
    ctx.fillStyle = `rgba(${hexToRgb(info.color)}, 0.15)`;
    ctx.fillRect(box.x, box.y + box.height + 2, box.width, 4);
    ctx.fillStyle = info.color;
    ctx.fillRect(box.x, box.y + box.height + 2, box.width * conf, 4);
    ctx.restore();

    // Landmarks (dots) when that filter is active
    if (activeFilter === 'landmarks' || activeFilter === 'neon') {
      drawLandmarks(ctx, det.landmarks.positions, info.color, activeFilter === 'neon');
    }
  });
}

function drawCornerAccents(ctx, box, color) {
  const len = 16;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.shadowBlur = 12;
  ctx.shadowColor = color;

  const corners = [
    [box.x, box.y, len, 0, 0, len],
    [box.x + box.width, box.y, -len, 0, 0, len],
    [box.x, box.y + box.height, len, 0, 0, -len],
    [box.x + box.width, box.y + box.height, -len, 0, 0, -len],
  ];

  corners.forEach(([x, y, dx1, dy1, dx2, dy2]) => {
    ctx.beginPath();
    ctx.moveTo(x + dx1, y + dy1);
    ctx.lineTo(x, y);
    ctx.lineTo(x + dx2, y + dy2);
    ctx.stroke();
  });
  ctx.restore();
}

function drawLandmarks(ctx, points, color, neon) {
  if (!points) return;
  ctx.save();
  if (neon) {
    ctx.shadowBlur = 10;
    ctx.shadowColor = color;
  }
  points.forEach(pt => {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, neon ? 2 : 1.5, 0, Math.PI * 2);
    ctx.fillStyle = neon ? color : `rgba(${hexToRgb(color)}, 0.8)`;
    ctx.fill();
  });
  ctx.restore();
}

// ────────────────────────────────────────────────────────────────
//  FILTERS / AR EFFECTS
// ────────────────────────────────────────────────────────────────
function drawFilter(ctx, detections, filter) {
  detections.forEach(det => {
    const lm = det.landmarks;
    if (!lm) return;

    switch (filter) {
      case 'sunglasses': drawSunglasses(ctx, lm); break;
      case 'dog':        drawDogFilter(ctx, lm, det.detection.box); break;
      case 'rainbow':    drawRainbow(ctx, lm, det.detection.box); break;
      case 'pixelate':   drawPixelate(ctx, det.detection.box); break;
      case 'neon':       drawNeonMesh(ctx, lm); break;
      case 'expression': drawExpressionFilter(ctx, lm, det.detection.box, det.expressions); break;
      default: break;
    }
  });
}

function drawSunglasses(ctx, lm) {
  const leftEye  = lm.getLeftEye();
  const rightEye = lm.getRightEye();
  if (!leftEye.length || !rightEye.length) return;

  const lc = centroid(leftEye);
  const rc = centroid(rightEye);
  const dist = Math.hypot(rc.x - lc.x, rc.y - lc.y);
  const size = dist * 0.65;

  ctx.save();
  ctx.strokeStyle = '#1a1a2e';
  ctx.lineWidth = 3;

  // Left lens
  ctx.beginPath();
  ctx.ellipse(lc.x, lc.y, size * 0.55, size * 0.4, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(20, 20, 50, 0.82)';
  ctx.fill();
  ctx.strokeStyle = '#7c3aed';
  ctx.stroke();

  // Right lens
  ctx.beginPath();
  ctx.ellipse(rc.x, rc.y, size * 0.55, size * 0.4, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Bridge
  ctx.beginPath();
  ctx.moveTo(lc.x + size * 0.55, lc.y);
  ctx.lineTo(rc.x - size * 0.55, rc.y);
  ctx.strokeStyle = '#7c3aed';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Gradient sheen
  const grad = ctx.createLinearGradient(lc.x, lc.y - size * 0.2, lc.x, lc.y);
  grad.addColorStop(0, 'rgba(167, 139, 250, 0.4)');
  grad.addColorStop(1, 'rgba(56, 189, 248, 0.15)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(lc.x, lc.y - size * 0.05, size * 0.45, size * 0.25, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(rc.x, rc.y - size * 0.05, size * 0.45, size * 0.25, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawDogFilter(ctx, lm, box) {
  const nose     = lm.getNose();
  const leftEye  = lm.getLeftEye();
  const rightEye = lm.getRightEye();
  const jaw      = lm.getJawOutline();

  if (!nose.length || !leftEye.length || !rightEye.length) return;

  const nc = centroid(nose);
  const lc = centroid(leftEye);
  const rc = centroid(rightEye);
  const dist = Math.hypot(rc.x - lc.x, rc.y - lc.y);

  // Dog nose
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(nc.x, nc.y, dist * 0.2, dist * 0.14, 0, 0, Math.PI * 2);
  ctx.fillStyle = '#2d1a00';
  ctx.fill();

  // Nose shine
  ctx.beginPath();
  ctx.ellipse(nc.x - dist * 0.06, nc.y - dist * 0.04, dist * 0.06, dist * 0.04, 0.4, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.fill();

  // Dog ears
  const topY = lm.positions.reduce((min, p) => Math.min(min, p.y), Infinity);

  // Left ear
  ctx.beginPath();
  ctx.ellipse(lc.x - dist * 0.2, topY - dist * 0.18, dist * 0.24, dist * 0.35, -0.3, 0, Math.PI * 2);
  ctx.fillStyle = '#c8833a';
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(lc.x - dist * 0.2, topY - dist * 0.1, dist * 0.15, dist * 0.24, -0.3, 0, Math.PI * 2);
  ctx.fillStyle = '#e8a85c';
  ctx.fill();

  // Right ear
  ctx.beginPath();
  ctx.ellipse(rc.x + dist * 0.2, topY - dist * 0.18, dist * 0.24, dist * 0.35, 0.3, 0, Math.PI * 2);
  ctx.fillStyle = '#c8833a';
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(rc.x + dist * 0.2, topY - dist * 0.1, dist * 0.15, dist * 0.24, 0.3, 0, Math.PI * 2);
  ctx.fillStyle = '#e8a85c';
  ctx.fill();

  // Freckles
  const freckleColor = 'rgba(139, 90, 30, 0.6)';
  [[-0.35, -0.02], [-0.28, 0.04], [-0.22, -0.05]].forEach(([dx, dy]) => {
    ctx.beginPath();
    ctx.arc(nc.x + dist * dx, nc.y + dist * dy, 3, 0, Math.PI * 2);
    ctx.fillStyle = freckleColor;
    ctx.fill();
  });
  [[0.35, -0.02], [0.28, 0.04], [0.22, -0.05]].forEach(([dx, dy]) => {
    ctx.beginPath();
    ctx.arc(nc.x + dist * dx, nc.y + dist * dy, 3, 0, Math.PI * 2);
    ctx.fillStyle = freckleColor;
    ctx.fill();
  });

  ctx.restore();
}

function drawRainbow(ctx, lm, box) {
  const leftEye  = lm.getLeftEye();
  const rightEye = lm.getRightEye();
  if (!leftEye.length || !rightEye.length) return;

  const lc = centroid(leftEye);
  const rc = centroid(rightEye);
  const cx = (lc.x + rc.x) / 2;
  const cy = Math.min(lc.y, rc.y) + 10;
  const radius = Math.hypot(rc.x - lc.x, rc.y - lc.y) * 1.3;

  const colors = ['#ef4444','#f97316','#eab308','#22c55e','#3b82f6','#8b5cf6','#ec4899'];

  ctx.save();
  colors.forEach((color, i) => {
    const r = radius - i * 8;
    if (r <= 0) return;
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = 6;
    ctx.globalAlpha = 0.75;
    ctx.stroke();
  });
  ctx.restore();
}

function drawPixelate(ctx, box) {
  // Pixelate the face region
  const pixSize = 14;
  const x = Math.round(box.x);
  const y = Math.round(box.y);
  const w = Math.round(box.width);
  const h = Math.round(box.height);

  // Draw video frame to temp canvas, sample colors
  const tmp = document.createElement('canvas');
  tmp.width  = video.videoWidth;
  tmp.height = video.videoHeight;
  const tctx = tmp.getContext('2d');
  tctx.drawImage(video, 0, 0);

  ctx.save();
  for (let px = x; px < x + w; px += pixSize) {
    for (let py = y; py < y + h; py += pixSize) {
      const data = tctx.getImageData(px + pixSize / 2, py + pixSize / 2, 1, 1).data;
      ctx.fillStyle = `rgba(${data[0]},${data[1]},${data[2]},${data[3] / 255})`;
      ctx.fillRect(px, py, pixSize, pixSize);
    }
  }
  ctx.restore();
}

function drawNeonMesh(ctx, lm) {
  const pts = lm.positions;
  if (!pts || pts.length < 68) return;

  // Draw connections like a wireframe mesh
  const CONNECTIONS = [
    // Jaw
    [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],
    // Left eyebrow
    [17,18],[18,19],[19,20],[20,21],
    // Right eyebrow
    [22,23],[23,24],[24,25],[25,26],
    // Nose bridge
    [27,28],[28,29],[29,30],
    // Nose bottom
    [30,31],[31,32],[32,33],[33,34],[34,35],
    // Left eye
    [36,37],[37,38],[38,39],[39,40],[40,41],[41,36],
    // Right eye
    [42,43],[43,44],[44,45],[45,46],[46,47],[47,42],
    // Outer mouth
    [48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,48],
    // Inner mouth
    [60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,60],
  ];

  ctx.save();
  ctx.strokeStyle = '#a78bfa';
  ctx.lineWidth = 1.2;
  ctx.shadowBlur = 12;
  ctx.shadowColor = '#a78bfa';
  ctx.globalAlpha = 0.85;

  CONNECTIONS.forEach(([a, b]) => {
    if (!pts[a] || !pts[b]) return;
    ctx.beginPath();
    ctx.moveTo(pts[a].x, pts[a].y);
    ctx.lineTo(pts[b].x, pts[b].y);
    ctx.stroke();
  });

  // Dots
  pts.forEach(pt => {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2);
    ctx.fillStyle = '#38bdf8';
    ctx.shadowColor = '#38bdf8';
    ctx.fill();
  });

  ctx.restore();
}

function drawExpressionFilter(ctx, lm, box, expressions) {
  const dom = getDominantExpression(expressions);
  switch (dom.key) {
    case 'happy':     drawRainbow(ctx, lm, box); break;
    case 'surprised': drawStarBurst(ctx, box); break;
    case 'sad':       drawRainDrops(ctx, lm, box); break;
    case 'angry':     drawFireEffect(ctx, lm, box); break;
    default:          drawNeonMesh(ctx, lm); break;
  }
}

function drawStarBurst(ctx, box) {
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const count = 12;
  ctx.save();
  ctx.globalAlpha = 0.7;
  for (let i = 0; i < count; i++) {
    const angle = (i / count) * Math.PI * 2;
    const x = cx + Math.cos(angle) * (box.width * 0.7);
    const y = cy + Math.sin(angle) * (box.height * 0.7);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.strokeStyle = `hsl(${(i / count) * 360}, 100%, 65%)`;
    ctx.lineWidth = 3;
    ctx.shadowBlur = 16;
    ctx.shadowColor = ctx.strokeStyle;
    ctx.stroke();
    // Star at end
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fill();
  }
  ctx.restore();
}

function drawRainDrops(ctx, lm, box) {
  const eyes = [...lm.getLeftEye(), ...lm.getRightEye()];
  if (!eyes.length) return;
  const eyeY = centroid(eyes).y;
  ctx.save();
  ctx.fillStyle = '#60a5fa';
  ctx.globalAlpha = 0.65;
  ctx.shadowBlur = 10;
  ctx.shadowColor = '#60a5fa';
  for (let i = 0; i < 6; i++) {
    const x = box.x + (box.width / 5) * i;
    const h = 20 + Math.random() * 25;
    const ys = eyeY + 10 + Math.random() * 15;
    ctx.beginPath();
    ctx.moveTo(x, ys);
    ctx.bezierCurveTo(x - 4, ys + h * 0.5, x - 4, ys + h * 0.8, x, ys + h);
    ctx.bezierCurveTo(x + 4, ys + h * 0.8, x + 4, ys + h * 0.5, x, ys);
    ctx.fill();
  }
  ctx.restore();
}

function drawFireEffect(ctx, lm, box) {
  const topY = lm.positions.reduce((min, p) => Math.min(min, p.y), Infinity);
  const cx   = box.x + box.width / 2;
  ctx.save();
  const grad = ctx.createRadialGradient(cx, topY, 5, cx, topY, box.width * 0.6);
  grad.addColorStop(0, 'rgba(255, 200, 50, 0.9)');
  grad.addColorStop(0.4, 'rgba(255, 90, 20, 0.7)');
  grad.addColorStop(1, 'rgba(220, 20, 0, 0)');
  ctx.fillStyle = grad;
  ctx.globalAlpha = 0.75;
  for (let i = 0; i < 5; i++) {
    const dx = (i - 2) * box.width * 0.15;
    ctx.beginPath();
    ctx.ellipse(cx + dx, topY - 20, box.width * 0.1, box.height * 0.25, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

// ────────────────────────────────────────────────────────────────
//  EXPRESSION UI
// ────────────────────────────────────────────────────────────────
function updateExpressionUI(detections) {
  if (!detections.length) {
    resetExpressionUI();
    return;
  }

  // Use first detected face
  const expr = detections[0].expressions;
  const sorted = Object.entries(expr).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([key, val]) => {
    const bar = document.getElementById(`bar-${key}`);
    const pct = document.getElementById(`pct-${key}`);
    if (bar) bar.style.width = `${val * 100}%`;
    if (pct) pct.textContent = `${Math.round(val * 100)}%`;

    const item = document.querySelector(`[data-expr="${key}"]`);
    if (item) item.classList.toggle('dominant', key === sorted[0][0]);
  });

  // Update dominant
  const [domKey, domVal] = sorted[0];
  const info = EXPRESSION_MAP[domKey] || EXPRESSION_MAP.neutral;
  dominantEmoji.textContent = info.emoji;
  dominantLabel.textContent = info.label;
  dominantSub.textContent   = `${Math.round(domVal * 100)}% de confiança`;
}

function resetExpressionUI() {
  Object.keys(EXPRESSION_MAP).forEach(key => {
    const bar = document.getElementById(`bar-${key}`);
    const pct = document.getElementById(`pct-${key}`);
    if (bar) bar.style.width = '0%';
    if (pct) pct.textContent = '0%';
    const item = document.querySelector(`[data-expr="${key}"]`);
    if (item) item.classList.remove('dominant');
  });
  dominantEmoji.textContent = '🎭';
  dominantLabel.textContent = 'Aguardando detecção';
  dominantSub.textContent   = 'Aponte seu rosto para a câmera';
}

// ────────────────────────────────────────────────────────────────
//  STATS
// ────────────────────────────────────────────────────────────────
function updateStats(detections, latency) {
  statFaces.textContent   = detections.length;
  statLatency.textContent = `${latency}ms`;

  if (detections.length) {
    const conf = Math.round(detections[0].detection.score * 100);
    statConf.textContent = `${conf}%`;
  } else {
    statConf.textContent = '--%';
  }
}

// ────────────────────────────────────────────────────────────────
//  HELPERS
// ────────────────────────────────────────────────────────────────
function getDominantExpression(expressions) {
  return Object.entries(expressions)
    .map(([key, value]) => ({ key, value }))
    .reduce((a, b) => a.value > b.value ? a : b);
}

function centroid(points) {
  const n = points.length;
  return {
    x: points.reduce((s, p) => s + p.x, 0) / n,
    y: points.reduce((s, p) => s + p.y, 0) / n,
  };
}

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r},${g},${b}`;
}

let toastTimer;
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3000);
}

// ────────────────────────────────────────────────────────────────
//  EVENT LISTENERS
// ────────────────────────────────────────────────────────────────

// Start button (first time)
startBtn.addEventListener('click', () => {
  startOverlay.style.display = 'none';
  toggleBtn.disabled = false;
  startCamera();
});

// Toggle camera on/off
toggleBtn.addEventListener('click', () => {
  if (stream) {
    stopCamera();
  } else {
    startCamera();
  }
});

// Mirror
mirrorBtn.addEventListener('click', () => {
  mirrored = !mirrored;
  applyMirror();
  mirrorBtn.style.color = mirrored ? 'var(--primary)' : '';
  showToast(mirrored ? '🪞 Espelho ativado' : '🪞 Espelho desativado');
});

// Filter buttons
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeFilter = btn.dataset.filter;
    if (filterCtx) filterCtx.clearRect(0, 0, filterCanvas.width, filterCanvas.height);
    showToast(`✨ Filtro: ${btn.querySelector('span:last-child').textContent}`);
  });
});

// Snapshot
snapshotBtn.addEventListener('click', () => {
  // Merge video + canvases onto a single canvas
  const tmp = document.createElement('canvas');
  tmp.width  = video.videoWidth;
  tmp.height = video.videoHeight;
  const tctx = tmp.getContext('2d');

  // Flip if mirrored
  if (mirrored) {
    tctx.translate(tmp.width, 0);
    tctx.scale(-1, 1);
  }
  tctx.drawImage(video, 0, 0);
  if (mirrored) { tctx.setTransform(1, 0, 0, 1, 0, 0); }

  tctx.drawImage(filterCanvas, 0, 0);
  tctx.drawImage(overlayCanvas, 0, 0);

  const dataUrl = tmp.toDataURL('image/png');
  snapshotImg.src = dataUrl;
  snapshotDl.href = dataUrl;
  snapshotModal.style.display = 'flex';
});

snapshotClose.addEventListener('click', () => {
  snapshotModal.style.display = 'none';
});

snapshotModal.addEventListener('click', e => {
  if (e.target === snapshotModal) snapshotModal.style.display = 'none';
});

// Resize handler
window.addEventListener('resize', () => {
  if (video.videoWidth) resizeCanvases();
});

// ────────────────────────────────────────────────────────────────
//  INIT
// ────────────────────────────────────────────────────────────────
(async () => {
  // Wait for face-api.js to load
  let attempts = 0;
  while (typeof faceapi === 'undefined' && attempts < 50) {
    await new Promise(r => setTimeout(r, 100));
    attempts++;
  }

  if (typeof faceapi === 'undefined') {
    loadingOverlay.querySelector('.loader-text').textContent = 'face-api.js não carregou';
    loadingOverlay.querySelector('.loader-sub').textContent  = 'Verifique sua conexão e recarregue a página.';
    return;
  }

  await loadModels();
})();
