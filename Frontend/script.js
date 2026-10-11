/*
 * SafeRide AI — standalone browser prototype
 *
 * Uses:
 *   - Browser getUserMedia() for live camera
 *   - MediaPipe Face Landmarker for facial landmarks
 *   - EAR for eye closure
 *   - MAR for mouth opening/yawn indication
 *   - Approximate head orientation
 *   - Escalating warnings
 *   - Critical full-browser alarm at 8 seconds of continuous eye closure
 *
 * Run with a local server:
 *   python -m http.server 8000
 * Then open:
 *   http://127.0.0.1:8000/Frontend/index.html
 */

import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MP_WASM =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const FACE_MODEL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const video = document.getElementById("cam-video");
const canvas = document.getElementById("landmark-canvas");
const ctx = canvas.getContext("2d");

const bookingBtn = document.getElementById("booking-btn");
const modeBadge = document.getElementById("mode-badge");
const modelStatus = document.getElementById("model-status");
const camFallback = document.getElementById("cam-fallback");
const camMsg = document.getElementById("cam-msg");
const overlayMetrics = document.getElementById("overlay-metrics");
const liveStatus = document.getElementById("live-status");
const liveDot = document.getElementById("live-dot");
const liveText = document.getElementById("live-text");

const criticalAlert = document.getElementById("critical-alert");
const criticalDuration = document.getElementById("critical-duration");
const dismissCriticalBtn = document.getElementById("dismiss-critical");

const closureTime = document.getElementById("closure-time");
const closureBar = document.getElementById("closure-bar");

const API_BASE_URL = "http://127.0.0.1:8000";
let DRIVER_ID = null;

let faceLandmarker = null;
let cameraStream = null;
let bookingActive = false;
let tripStarting = false;
let processing = false;
let animationId = null;
let cameraWatchdog = null;
let lastVideoTime = -1;
let frameCounter = 0;
let inferenceCounter = 0;
let lastFpsTime = performance.now();
let aiFps = 0;

let eyesClosedSince = null;
let criticalTriggered = false;
let warningLevel = 0;
let lastAlertAt = 0;
let lastYawnAt = 0;

// Yawn tracking: one yawn is counted only after the mouth opens long enough
// and then closes again. This prevents one long yawn from being counted repeatedly.
let yawnCount = 0;
let yawnInProgress = false;
let yawnWarningLevel = 0;
let refreshRecommendationShown = false;

// Distraction tracking: continuous non-forward head orientation.
let distractionSince = null;
let distractionWarningLevel = 0;
let distractionCriticalTriggered = false;

let audioContext = null;
let alarmTimer = null;
let speechTimer = null;

let alertness = 98;
// ============================================================
// PHASE 1 — RISK ENGINE STATE
// ============================================================

// Smoothed facial signals.
// These reduce frame-to-frame MediaPipe landmark noise.
let smoothedEAR = null;
let smoothedMAR = null;

// Stable eye state with hysteresis.
let stableEyesClosed = false;

// Stable head state.
// A short head movement is not immediately treated as distraction.
let stableHeadState = "Stable";
let headCandidateState = "Stable";
let headCandidateSince = null;

// Face tracking protection.
// Temporary face-loss should not immediately create a safety event.
let noFaceSince = null;

// Unified driver safety state.
let currentRiskLevel = "SAFE";
let currentRiskScore = 98;
let currentRiskReasons = [];

// Prototype calibration values.
// These are NOT clinically validated thresholds.
const SIGNAL_SMOOTHING_ALPHA = 0.25;

const EAR_OPEN_RECOVERY = 0.225;

// Head orientation must remain in a new state for this long
// before the state is accepted.
const HEAD_STABILITY_MS = 350;

// Ignore very short MediaPipe face-tracking gaps.
const FACE_LOSS_GRACE_MS = 700;

// Trip report state. The report is kept in localStorage after a trip ends
// so the prototype visibly demonstrates trip-data persistence in the browser.
let tripStartTime = null;
let tripElapsedMs = 0;
let maxEyeClosureMs = 0;
let tripWarningCount = 0;
let distractionIncidentCount = 0;
let tripCriticalCount = 0;
let tripEventCount = 0;
let reportTimer = null;
let lastSavedTrip = null;

const CRITICAL_EYE_CLOSURE_MS = 8000;

// EAR thresholds are prototype calibration values.
// They should be calibrated for the camera/driver population before real deployment.
const EAR_CLOSED = 0.205;
const EAR_BLINK_GRACE_MS = 450;

const YAWN_MAR = 0.42;
const YAWN_HOLD_MS = 700;

const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

let mouthOpenSince = null;

function $(id) {
    return document.getElementById(id);
}

function setModelStatus(text, state) {
    modelStatus.textContent = text;
    modelStatus.className = "model-status " + (state || "");
}

function updateClock() {
    $("clock").textContent = new Date().toLocaleTimeString("en-IN", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
    });
}
setInterval(updateClock, 1000);
updateClock();

function distance(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = (a.z || 0) - (b.z || 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function eyeAspectRatio(lm, ids) {
    const p1 = lm[ids[0]];
    const p2 = lm[ids[1]];
    const p3 = lm[ids[2]];
    const p4 = lm[ids[3]];
    const p5 = lm[ids[4]];
    const p6 = lm[ids[5]];

    if (!p1 || !p2 || !p3 || !p4 || !p5 || !p6) return 0;

    const vertical1 = distance(p2, p6);
    const vertical2 = distance(p3, p5);
    const horizontal = distance(p1, p4);

    if (horizontal === 0) return 0;

    return (vertical1 + vertical2) / (2 * horizontal);
}

function mouthAspectRatio(lm) {
    const top = lm[13];
    const bottom = lm[14];
    const left = lm[78];
    const right = lm[308];

    if (!top || !bottom || !left || !right) return 0;

    const vertical = distance(top, bottom);
    const horizontal = distance(left, right);

    if (horizontal === 0) return 0;

    return vertical / horizontal;
}
// ============================================================
// PHASE 1 — SIGNAL SMOOTHING
// ============================================================

function smoothSignal(previous, current, alpha = SIGNAL_SMOOTHING_ALPHA) {
    if (!Number.isFinite(current)) {
        return previous;
    }

    if (previous === null || !Number.isFinite(previous)) {
        return current;
    }

    return previous + alpha * (current - previous);
}


// ============================================================
// STABLE EYE STATE
// ============================================================

function getStableEyeState(ear) {
    if (!Number.isFinite(ear) || ear <= 0) {
        return stableEyesClosed;
    }

    // Once the eyes are considered closed, require a clearly
    // higher EAR to mark them open again.
    if (stableEyesClosed) {
        if (ear >= EAR_OPEN_RECOVERY) {
            stableEyesClosed = false;
        }
    } else {
        if (ear <= EAR_CLOSED) {
            stableEyesClosed = true;
        }
    }

    return stableEyesClosed;
}


// ============================================================
// STABLE HEAD STATE
// ============================================================

function getStableHeadState(rawState, now) {

    // Unknown should not immediately become a distraction event.
    if (!rawState || rawState === "Unknown") {
        return stableHeadState;
    }

    // Nothing changed.
    if (rawState === stableHeadState) {
        headCandidateState = rawState;
        headCandidateSince = null;
        return stableHeadState;
    }

    // New candidate state detected.
    if (rawState !== headCandidateState) {
        headCandidateState = rawState;
        headCandidateSince = now;
        return stableHeadState;
    }

    // Candidate must persist before becoming the official state.
    if (
        headCandidateSince &&
        now - headCandidateSince >= HEAD_STABILITY_MS
    ) {
        stableHeadState = headCandidateState;
        headCandidateSince = null;
    }

    return stableHeadState;
}

/*
 * This is a lightweight prototype head-orientation estimate.
 * It is intentionally not presented as a medically/industrially validated pose estimator.
 */
function estimateHeadState(lm) {
    const leftEyeOuter = lm[33];
    const rightEyeOuter = lm[263];
    const nose = lm[1];
    const forehead = lm[10];
    const chin = lm[152];

    if (!leftEyeOuter || !rightEyeOuter || !nose || !forehead || !chin) {
        return { label: "Unknown", score: 0 };
    }

    const eyeCenterX = (leftEyeOuter.x + rightEyeOuter.x) / 2;
    const eyeWidth = Math.abs(rightEyeOuter.x - leftEyeOuter.x) || 0.001;

    const yawRatio = (nose.x - eyeCenterX) / eyeWidth;

    const eyeCenterY = (leftEyeOuter.y + rightEyeOuter.y) / 2;
    const faceHeight = Math.abs(chin.y - forehead.y) || 0.001;
    const pitchRatio = (nose.y - eyeCenterY) / faceHeight;

    if (Math.abs(yawRatio) > 0.32) {
        return { label: "Turned Away", score: 1 };
    }

    if (pitchRatio > 0.25) {
        return { label: "Downward", score: 1 };
    }

    return { label: "Stable", score: 0 };
}

function drawLandmarks(lm) {
    if (!canvas.width || !canvas.height) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;

    // Draw only useful landmarks so the UI remains clean.
    const useful = [
        ...LEFT_EYE,
        ...RIGHT_EYE,
        13, 14, 78, 308,
        1, 10, 152
    ];

    ctx.fillStyle = "#22c55e";

    for (const index of useful) {
        const p = lm[index];
        if (!p) continue;

        // Video is mirrored using CSS, so mirror x for the overlay.
        const x = (1 - p.x) * w;
        const y = p.y * h;

        ctx.beginPath();
        ctx.arc(x, y, 2.3, 0, Math.PI * 2);
        ctx.fill();
    }

    // Eye lines
    ctx.strokeStyle = "rgba(34,197,94,.75)";
    ctx.lineWidth = 1.5;

    function line(a, b) {
        const p1 = lm[a], p2 = lm[b];
        if (!p1 || !p2) return;

        ctx.beginPath();
        ctx.moveTo((1 - p1.x) * w, p1.y * h);
        ctx.lineTo((1 - p2.x) * w, p2.y * h);
        ctx.stroke();
    }

    line(33, 160);
    line(160, 158);
    line(158, 133);
    line(133, 153);
    line(153, 144);
    line(144, 33);

    line(362, 385);
    line(385, 387);
    line(387, 263);
    line(263, 373);
    line(373, 380);
    line(380, 362);
}

function updateFaceBox(lm) {
    const xs = [];
    const ys = [];

    for (const p of lm) {
        if (!p) continue;
        xs.push(p.x);
        ys.push(p.y);
    }

    if (!xs.length) return;

    const minX = Math.max(0, Math.min(...xs));
    const maxX = Math.min(1, Math.max(...xs));
    const minY = Math.max(0, Math.min(...ys));
    const maxY = Math.min(1, Math.max(...ys));

    const box = $("face-box");

    // Because the video is mirrored, mirror the x position too.
    const left = (1 - maxX) * 100;
    const width = (maxX - minX) * 100;
    const top = minY * 100;
    const height = (maxY - minY) * 100;

    box.style.left = left + "%";
    box.style.top = top + "%";
    box.style.width = width + "%";
    box.style.height = height + "%";
    box.style.display = "block";
}

function setIndicator(id, color, pulse = false) {
    $(id).className =
        "w-3 h-3 rounded-full " + color + (pulse ? " pulse-alert" : "");
}

function updateMetrics(ear, mar, head) {
    $("ear-val").textContent = ear.toFixed(3);
    $("mar-val").textContent = mar.toFixed(3);
    $("head-val").textContent = head;
    $("fps-val").textContent = aiFps.toFixed(0);
}

function setLiveState(text, colorClass) {
    liveText.textContent = text;
    liveDot.className = "status-dot " + colorClass + " mr-2";
}
// =========================================================
// PHASE 2 — LIVE RISK UI
// =========================================================

function updateRiskUI(level, score, reasons = []) {

    const panel = $("risk-panel");
    const levelEl = $("risk-level");
    const scoreEl = $("risk-score");
    const barEl = $("risk-bar");
    const messageEl = $("risk-message");
    const reasonsEl = $("risk-reasons");
    const signalCountEl = $("risk-signal-count");
    const iconEl = $("risk-icon");

    if (!panel || !levelEl || !scoreEl || !barEl ||
        !messageEl || !reasonsEl || !signalCountEl || !iconEl) {
        return;
    }

    const safeScore = Math.max(
        0,
        Math.min(100, Math.round(score ?? 0))
    );

    const normalizedLevel = String(level || "UNKNOWN")
        .toUpperCase();

    // -----------------------------------------------------
    // RESET STATE CLASSES
    // -----------------------------------------------------

    panel.classList.remove(
        "risk-state-safe",
        "risk-state-low",
        "risk-state-high",
        "risk-state-critical",
        "risk-state-unknown"
    );

    iconEl.classList.remove(
        "risk-safe",
        "risk-low",
        "risk-high",
        "risk-critical",
        "risk-unknown"
    );

    // -----------------------------------------------------
    // SAFE
    // -----------------------------------------------------

    if (normalizedLevel === "SAFE") {

        panel.classList.add("risk-state-safe");
        iconEl.classList.add("risk-safe");

        levelEl.textContent = "SAFE";
        levelEl.className =
            "text-2xl font-black mt-1 text-emerald-400";

        iconEl.textContent = "✓";

        barEl.style.width = safeScore + "%";
        barEl.style.background = "#10b981";

        scoreEl.textContent = safeScore + " / 100";

        messageEl.className =
            "mt-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10 px-3 py-3";

        messageEl.innerHTML = `
            <p class="text-xs text-emerald-300 font-medium">
                No active safety concerns.
            </p>
        `;
    }

    // -----------------------------------------------------
    // LOW RISK
    // -----------------------------------------------------

    else if (normalizedLevel === "LOW RISK") {

        panel.classList.add("risk-state-low");
        iconEl.classList.add("risk-low");

        levelEl.textContent = "LOW RISK";
        levelEl.className =
            "text-2xl font-black mt-1 text-amber-400";

        iconEl.textContent = "!";

        barEl.style.width = safeScore + "%";
        barEl.style.background = "#f59e0b";

        scoreEl.textContent = safeScore + " / 100";

        messageEl.className =
            "mt-4 rounded-xl bg-amber-500/5 border border-amber-500/10 px-3 py-3";

        messageEl.innerHTML = `
            <p class="text-xs text-amber-300 font-medium">
                Minor safety signals detected. Stay attentive.
            </p>
        `;
    }

    // -----------------------------------------------------
    // HIGH RISK
    // -----------------------------------------------------

    else if (normalizedLevel === "HIGH RISK") {

        panel.classList.add("risk-state-high");
        iconEl.classList.add("risk-high");

        levelEl.textContent = "HIGH RISK";
        levelEl.className =
            "text-2xl font-black mt-1 text-orange-400";

        iconEl.textContent = "!";

        barEl.style.width = safeScore + "%";
        barEl.style.background = "#f97316";

        scoreEl.textContent = safeScore + " / 100";

        messageEl.className =
            "mt-4 rounded-xl bg-orange-500/5 border border-orange-500/15 px-3 py-3";

        messageEl.innerHTML = `
            <p class="text-xs text-orange-300 font-medium">
                Attention required. One or more safety signals are elevated.
            </p>
        `;
    }

    // -----------------------------------------------------
    // CRITICAL
    // -----------------------------------------------------

    else if (normalizedLevel === "CRITICAL") {

        panel.classList.add("risk-state-critical");
        iconEl.classList.add("risk-critical");

        levelEl.textContent = "CRITICAL";
        levelEl.className =
            "text-2xl font-black mt-1 text-red-400";

        iconEl.textContent = "⚠";

        barEl.style.width = safeScore + "%";
        barEl.style.background = "#ef4444";

        scoreEl.textContent = safeScore + " / 100";

        messageEl.className =
            "mt-4 rounded-xl bg-red-500/10 border border-red-500/25 px-3 py-3";

        messageEl.innerHTML = `
            <p class="text-xs text-red-300 font-bold">
                Immediate driver attention required.
            </p>
        `;
    }

    // -----------------------------------------------------
    // UNKNOWN / NO FACE
    // -----------------------------------------------------

    else {

        panel.classList.add("risk-state-unknown");
        iconEl.classList.add("risk-unknown");

        levelEl.textContent = "NO FACE";
        levelEl.className =
            "text-2xl font-black mt-1 text-gray-400";

        iconEl.textContent = "—";

        barEl.style.width = "0%";
        barEl.style.background = "#6b7280";

        scoreEl.textContent = "-- / 100";

        messageEl.className =
            "mt-4 rounded-xl bg-gray-500/5 border border-gray-500/10 px-3 py-3";

        messageEl.innerHTML = `
            <p class="text-xs text-gray-400 font-medium">
                Driver face is not currently visible to the camera.
            </p>
        `;
    }

    // -----------------------------------------------------
    // ACTIVE SIGNALS
    // -----------------------------------------------------

    signalCountEl.textContent = reasons.length;

    if (!reasons.length) {

        reasonsEl.innerHTML = `
            <div class="risk-reason safe">
                <span class="risk-reason-icon">✓</span>
                <span>No active risk signals</span>
            </div>
        `;

        return;
    }

    reasonsEl.innerHTML = reasons
        .slice(0, 4)
        .map(reason => {

            const severity = reason.severity || "low";

            let icon = "!";

            if (severity === "critical") {
                icon = "⚠";
            } else if (severity === "high") {
                icon = "!";
            } else if (severity === "medium") {
                icon = "•";
            } else {
                icon = "·";
            }

            return `
                <div class="risk-reason ${severity}">
                    <span class="risk-reason-icon">${icon}</span>
                    <span>${reason.text}</span>
                </div>
            `;

        })
        .join("");
}

function updateGauge(value) {

    const val = Math.max(
        0,
        Math.min(
            100,
            Math.round(value)
        )
    );

    const circle =
        $("gauge-circle");

    const circumference = 327;

    const offset =
        circumference -
        (circumference * val / 100);

    circle.style.strokeDashoffset =
        offset;

    const status =
        $("score-status");

    const scoreText =
        $("score-text");

    scoreText.textContent =
        val;


    // ========================================================
    // RISK LEVEL
    // ========================================================

    let level = "SAFE";

    if (val >= 85) {

        level = "SAFE";

        circle.style.stroke =
            "#10b981";

        status.textContent =
            "SAFE";

        status.className =
            "text-emerald-400 text-sm font-medium mt-3";

    } else if (val >= 70) {

        level = "LOW RISK";

        circle.style.stroke =
            "#f59e0b";

        status.textContent =
            "LOW RISK";

        status.className =
            "text-amber-400 text-sm font-medium mt-3";

    } else if (val >= 50) {

        level = "HIGH RISK";

        circle.style.stroke =
            "#f97316";

        status.textContent =
            "HIGH RISK";

        status.className =
            "text-orange-400 text-sm font-medium mt-3 pulse-alert";

    } else {

        level = "CRITICAL";

        circle.style.stroke =
            "#ef4444";

        status.textContent =
            "CRITICAL";

        status.className =
            "text-red-400 text-sm font-medium mt-3 pulse-alert";
    }


    currentRiskLevel =
        level;

    currentRiskScore =
        val;
    intelligentIntervention();
}


let lastInterventionLevel = "";
let lastInterventionTime = 0;
let highRiskStartTime = 0;
let criticalEscalated = false;

function intelligentIntervention() {
    if (!bookingActive) return;

    const level = String(currentRiskLevel || "").toUpperCase();
    const now = Date.now();
    if (level.includes("HIGH")) {
    if (highRiskStartTime === 0) {
        highRiskStartTime = now;
    }

    if (now - highRiskStartTime >= 15000 && !criticalEscalated) {
        criticalEscalated = true;
        currentRiskLevel = "CRITICAL";
        currentRiskScore = Math.min(currentRiskScore, 49);
    }
} else {
    highRiskStartTime = 0;
    criticalEscalated = false;
}

    if (!level || level === "SAFE") {
        lastInterventionLevel = "";
        highRiskStartTime = 0;
        criticalEscalated = false;
        return;
    }

    // Avoid repeating the same intervention too frequently.
    if (
        level === lastInterventionLevel &&
        now - lastInterventionTime < 10000 &&
        !level.includes("CRITICAL")
    ) {
        return;
    }

    lastInterventionLevel = level;
    lastInterventionTime = now;

    if (criticalEscalated) {
        currentRiskLevel = "CRITICAL";
        currentRiskScore = Math.min(currentRiskScore, 49);
    }

    const interventionLevel = String(currentRiskLevel || level).toUpperCase();
if (interventionLevel.includes("CRITICAL")) {
    addAlert(
        "🚨 CRITICAL FATIGUE: Pull over safely immediately. Stop driving and rest before continuing.",
        "critical"
    );

    const safeStopButton = document.createElement("button");
    safeStopButton.textContent = "📍 Find a Safe Stop";
    safeStopButton.className =
        "mt-2 px-4 py-2 rounded-lg bg-emerald-500 text-black font-semibold";

    safeStopButton.onclick = () => {
        const choice = prompt(
            "Choose a safe stop:\n1. Petrol Pumps\n2. Hospitals\n3. Police Stations\n4. Rest Areas",
            "4"
        );

        const categories = {
            "1": "petrol",
            "2": "hospital",
            "3": "police",
            "4": "rest"
        };

        if (!categories[choice]) return;

        const url =
            "http://127.0.0.1:8000/safe-stops?category=" +
            encodeURIComponent(categories[choice]) +
            "&latitude=28.6692&longitude=77.4538";

        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error("Safe Stop API request failed");
                }
                return response.json();
            })
            .then(data => {
                if (data.success && data.maps_url) {
                    window.open(
                        data.maps_url,
                        "_blank",
                        "noopener,noreferrer"
                    );
                } else {
                    alert(data.message || "No safe-stop link available.");
                }
            })
            .catch(error => {
                console.error("Safe Stop error:", error);
                alert("Safe Stop API unavailable. Check the backend.");
            });
    };

        
    const alertContainer = document.querySelector("#alert-container");
    if (alertContainer && !alertContainer.querySelector(".safe-stop-button")) {
        safeStopButton.classList.add("safe-stop-button");
        alertContainer.appendChild(safeStopButton);
    }



    } else if (level.includes("HIGH")) {
        addAlert(
            "⚠️ HIGH RISK: Avoid phone use, focus on the road, and stop at a safe location if fatigue continues.",
            "warning"
        );
    } else if (level.includes("LOW")) {
        addAlert(
            "ℹ️ LOW RISK: Stay focused, avoid unnecessary phone use, and maintain a safe following distance.",
            "warning"
        );
    }
}


function addAlert(message, severity = "warning") {
    if (bookingActive && severity !== "normal") {
        tripWarningCount++;
        tripEventCount++;
    } else if (bookingActive && severity === "normal") {
        tripEventCount++;
    }

    const log = $("alert-log");

    const empty = log.querySelector("p.italic");
    if (empty) empty.remove();

    const div = document.createElement("div");

    const classes = severity === "critical"
        ? "bg-red-500/20 border-red-500/50 text-red-200"
        : severity === "caution"
        ? "bg-amber-500/10 border-amber-500/30 text-amber-200"
        : "bg-emerald-500/10 border-emerald-500/20 text-emerald-200";

    div.className =
        "text-xs border rounded-lg px-3 py-2 " + classes;

    div.textContent =
        new Date().toLocaleTimeString("en-IN", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        }) + " — " + message;

    log.prepend(div);
    if (severity === "critical") {
    const safeStopButton = document.createElement("button");
    safeStopButton.textContent = "📍 Find a Safe Stop";
    safeStopButton.className =
        "mt-2 px-3 py-2 rounded-lg bg-emerald-500 text-black font-semibold";

    safeStopButton.onclick = () => {
        const choice = prompt(
            "Choose a safe stop:\n1. Petrol Pumps\n2. Hospitals\n3. Police Stations\n4. Rest Areas",
            "4"
        );

        const categories = {
            "1": "petrol",
            "2": "hospital",
            "3": "police",
            "4": "rest"
        };

        if (!categories[choice]) return;

        const url =
            "http://127.0.0.1:8000/safe-stops?category=" +
            encodeURIComponent(categories[choice]) +
            "&latitude=28.6692&longitude=77.4538";

        fetch(url)
            .then(response => {
                if (!response.ok) throw new Error("API request failed");
                return response.json();
            })
            .then(data => {
                if (data.success && data.maps_url) {
                    window.open(data.maps_url, "_blank", "noopener,noreferrer");
                } else {
                    alert(data.message || "Safe stop unavailable.");
                }
            })
            .catch(() => alert("Safe Stop API unavailable. Check the backend."));
    };

    div.appendChild(safeStopButton);
}

    while (log.children.length > 10) {
        log.lastChild.remove();
    }
}

function ensureAudio() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    if (audioContext.state === "suspended") {
        audioContext.resume().catch(() => {});
    }

    return audioContext;
}

function playAlarmTone(frequency = 900, duration = 220) {
    try {
        const audio = ensureAudio();
        const osc = audio.createOscillator();
        const gain = audio.createGain();

        osc.type = "square";
        osc.frequency.value = frequency;

        gain.gain.setValueAtTime(0.001, audio.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.28, audio.currentTime + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + duration / 1000);

        osc.connect(gain);
        gain.connect(audio.destination);

        osc.start();
        osc.stop(audio.currentTime + duration / 1000 + 0.03);
    } catch (e) {
        console.warn("Alarm audio unavailable:", e);
    }
}

function startCriticalAlarm(speechText) {
    if (alarmTimer) return;

    playAlarmTone(1100, 250);

    alarmTimer = setInterval(() => {
        playAlarmTone(1100, 250);
        setTimeout(() => playAlarmTone(700, 250), 280);
    }, 900);

    speakCriticalAlert(speechText);
}

function stopCriticalAlarm() {
    if (alarmTimer) {
        clearInterval(alarmTimer);
        alarmTimer = null;
    }

    if (speechTimer) {
        clearTimeout(speechTimer);
        speechTimer = null;
    }

    try {
        window.speechSynthesis.cancel();
    } catch (e) {}
}

function speakCriticalAlert(speechText) {
    if (!("speechSynthesis" in window)) return;

    try {
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(
            speechText || "Critical driver alert. Please pay attention and stop safely as soon as possible."
        );

        utterance.rate = 0.95;
        utterance.pitch = 1.05;
        utterance.volume = 1.0;

        window.speechSynthesis.speak(utterance);

        speechTimer = setTimeout(() => {
            if (criticalTriggered) speakCriticalAlert();
        }, 6500);
    } catch (e) {
        console.warn("Speech unavailable:", e);
    }
}

function showCriticalAlert({
    title,
    message,
    durationLabel = "",
    instruction,
    speech
}) {
    if (criticalTriggered) return;

    criticalTriggered = true;
    if (bookingActive) {
        tripCriticalCount++;
        tripEventCount++;
    }

    $("critical-title").textContent = title;
    $("critical-message").textContent = message;
    $("critical-instruction").textContent = instruction;

    const durationBox = $("critical-duration-box");
    if (durationLabel) {
        durationBox.classList.remove("hidden");
        durationBox.textContent = durationLabel;
    } else {
        durationBox.classList.add("hidden");
    }

    criticalAlert.classList.remove("hidden");

    startCriticalAlarm(speech);
}

function hideCriticalAlert() {
    criticalTriggered = false;
    criticalAlert.classList.add("hidden");
    stopCriticalAlarm();
}

dismissCriticalBtn.addEventListener("click", () => {
    hideCriticalAlert();
    addAlert("Driver acknowledged the critical alert.", "caution");
});

function formatTripDuration(ms) {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return hours > 0
        ? String(hours).padStart(2, "0") + ":" + String(minutes).padStart(2, "0") + ":" + String(seconds).padStart(2, "0")
        : String(minutes).padStart(2, "0") + ":" + String(seconds).padStart(2, "0");
}

function calculateTripRating() {
    let score = 10;
    score -= Math.min(2.5, yawnCount * 0.35);
    score -= Math.min(2.5, distractionIncidentCount * 0.8);
    score -= Math.min(2.0, tripWarningCount * 0.15);
    score -= Math.min(2.5, maxEyeClosureMs >= 8000 ? 2.5 : maxEyeClosureMs >= 5000 ? 1.6 : maxEyeClosureMs >= 3000 ? 0.8 : maxEyeClosureMs >= 1000 ? 0.35 : 0);
    score -= Math.min(1.5, tripCriticalCount * 0.75);
    return Math.max(0, Math.min(10, score));
}

function starsForScore(score) {
    const rounded = Math.max(
        0,
        Math.min(5, Math.round(score / 2))
    );

    return "★".repeat(rounded) + "☆".repeat(5 - rounded);
}

function ratingText(score) {
    if (score >= 9) return "Excellent driving performance";
    if (score >= 7.5) return "Good performance — minor safety events detected";
    if (score >= 6) return "Needs improvement — fatigue/distraction events detected";
    if (score >= 4) return "Poor performance — frequent safety warnings";
    return "High-risk trip — immediate improvement required";
}

function updateTripReport(save = false) {
    if (tripStartTime) tripElapsedMs = performance.now() - tripStartTime;

    const score = calculateTripRating();
    const duration = formatTripDuration(tripElapsedMs);

    $("report-duration").textContent = duration;
    $("report-score").textContent = score.toFixed(1) + " / 10";
    $("report-stars").textContent = starsForScore(score);
    $("report-rating-text").textContent = ratingText(score);
    $("report-yawns").textContent = yawnCount;
    $("report-warnings").textContent = tripWarningCount;
    $("report-distractions").textContent = distractionIncidentCount;
    $("report-eye").textContent = (maxEyeClosureMs / 1000).toFixed(1) + "s";

    const progress = Math.max(0, Math.min(100, score * 10));
    $("report-progress").style.width = progress + "%";
    $("report-progress-label").textContent = Math.round(progress) + "%";
    $("report-progress").className =
        "h-full transition-all " + (score >= 7.5 ? "bg-emerald-500" : score >= 5 ? "bg-amber-500" : "bg-red-500");

    $("report-summary").textContent = tripStartTime
        ? `Live report: ${tripEventCount} safety events recorded. Keep your attention forward and take a break if fatigue increases.`
        : "Trip report generated from the completed safety session.";

    if (save) {
        const report = {
            id: "TRIP-" + Date.now(),
            completedAt: new Date().toISOString(),
            durationMs: tripElapsedMs,
            duration: duration,
            score: Number(score.toFixed(1)),
            stars: Math.round(score / 2),
            yawns: yawnCount,
            warnings: tripWarningCount,
            distractions: distractionIncidentCount,
            maxEyeClosureSeconds: Number((maxEyeClosureMs / 1000).toFixed(1)),
            criticalAlerts: tripCriticalCount,
            events: tripEventCount
        };

        localStorage.setItem("saferide_last_trip_report", JSON.stringify(report));
        lastSavedTrip = report;
        $("report-storage").textContent = "SAVED LOCALLY";
        $("report-storage").className = "text-[10px] px-2 py-1 rounded-full bg-emerald-500/10 text-emerald-400";
        $("report-state").textContent = "Trip completed • report stored in browser";
    }
}

function loadLastTripReport() {
    try {
        const raw = localStorage.getItem("saferide_last_trip_report");
        if (!raw) return;
        const report = JSON.parse(raw);
        if (!report) return;

        tripElapsedMs = report.durationMs || 0;
        yawnCount = report.yawns || 0;
        tripWarningCount = report.warnings || 0;
        distractionIncidentCount = report.distractions || 0;
        maxEyeClosureMs = (report.maxEyeClosureSeconds || 0) * 1000;
        tripCriticalCount = report.criticalAlerts || 0;
        tripEventCount = report.events || 0;
        updateTripReport(false);
        $("report-state").textContent = "Last completed trip • loaded from browser storage";
        $("report-storage").textContent = "SAVED LOCALLY";
        $("report-storage").className = "text-[10px] px-2 py-1 rounded-full bg-emerald-500/10 text-emerald-400";
    } catch (e) {
        console.warn("Could not load saved trip report", e);
    }
}

function startTripReportTimer() {
    clearInterval(reportTimer);
    reportTimer = setInterval(() => updateTripReport(false), 500);
}

function stopTripReportTimer() {
    clearInterval(reportTimer);
    reportTimer = null;
}

function recordTripEvent() {
    if (bookingActive) tripEventCount++;
}

function updateYawnUI() {
    const count = Math.min(yawnCount, 5);
    $("yawn-count").textContent = count + " / 5";
    $("yawn-bar").style.width = Math.min(100, (count / 5) * 100) + "%";

    const status = $("yawn-status");

    if (yawnCount >= 5) {
        $("yawn-bar").className = "h-full bg-red-500 transition-all";
        status.textContent = "Refresh advised";
        status.className = "text-red-400 text-sm font-bold pulse-alert";
    } else if (yawnCount >= 2) {
        $("yawn-bar").className = "h-full bg-amber-500 transition-all";
        status.textContent = "Warning";
        status.className = "text-amber-400 text-sm font-bold";
    } else {
        $("yawn-bar").className = "h-full bg-emerald-500 transition-all";
        status.textContent = "Normal";
        status.className = "text-emerald-400 text-sm font-bold";
    }
}

function showRefreshRecommendation() {
    if (refreshRecommendationShown) return;

    refreshRecommendationShown = true;
    $("refresh-message").textContent =
        "5 yawns detected during this trip. This may indicate fatigue.";
    $("refresh-alert").classList.remove("hidden");

    addAlert(
        "5 yawns detected — driver should stop safely and get refreshed.",
        "critical"
    );

    playAlarmTone(650, 250);

    if ("speechSynthesis" in window) {
        try {
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(
                "Fatigue warning. Five yawns detected. Please stop driving safely and take a short break to get refreshed."
            );
            u.rate = 0.9;
            u.volume = 1;
            window.speechSynthesis.speak(u);
        } catch (e) {}
    }
}

$("dismiss-refresh").addEventListener("click", () => {
    $("refresh-alert").classList.add("hidden");
    addAlert("Driver acknowledged the fatigue refresh recommendation.", "caution");
});

function registerYawn(now) {
    yawnCount++;
    lastYawnAt = now;

    updateYawnUI();

    if (yawnCount === 1) {
        addAlert("Yawn 1 detected.", "caution");
    } else if (yawnCount === 2 && yawnWarningLevel < 1) {
        yawnWarningLevel = 1;
        addAlert("Yawn warning — 2 yawns detected. Please stay alert.", "caution");
        $("alert-overlay").classList.remove("hidden");
        $("alert-overlay-text").textContent = "⚠  FATIGUE WARNING — 2 YAWNS DETECTED";
        playAlarmTone(720, 180);
    } else if (yawnCount > 2 && yawnCount < 5) {
        addAlert("Yawn " + yawnCount + " detected. Fatigue level is increasing.", "caution");
    }

    if (yawnCount >= 5) {
        showRefreshRecommendation();
    }
}

function updateDistractionUI(now) {
    const timeEl = $("distraction-time");
    const bar = $("distraction-bar");
    const status = $("distraction-status");

    if (!distractionSince) {
        timeEl.textContent = "0.0s";
        timeEl.className = "mono text-xl font-bold text-emerald-400";
        bar.style.width = "0%";
        bar.className = "h-full bg-emerald-500 transition-all";
        status.textContent = "Focused";
        status.className = "text-emerald-400 text-sm font-bold";
        return;
    }

    const elapsed = now - distractionSince;
    const seconds = elapsed / 1000;
    timeEl.textContent = seconds.toFixed(1) + "s";
    bar.style.width = Math.min(100, (elapsed / 15000) * 100) + "%";

    if (elapsed < 10000) {
        timeEl.className = "mono text-xl font-bold text-amber-400";
        bar.className = "h-full bg-amber-500 transition-all";
        status.textContent = "Attention drifting";
        status.className = "text-amber-400 text-sm font-bold";
    } else if (elapsed < 15000) {
        timeEl.className = "mono text-xl font-bold text-orange-400 pulse-alert";
        bar.className = "h-full bg-orange-500 transition-all";
        status.textContent = "Warning";
        status.className = "text-orange-400 text-sm font-bold";
    } else {
        timeEl.className = "mono text-xl font-bold text-red-400 pulse-alert";
        bar.className = "h-full bg-red-500 transition-all";
        status.textContent = "CRITICAL";
        status.className = "text-red-400 text-sm font-bold pulse-alert";
    }
}

function handleDistraction(headState, now) {
    const distracted = headState === "Turned Away" || headState === "Downward";

    if (!distracted) {
        const previous = distractionSince ? now - distractionSince : 0;
        distractionSince = null;
        distractionWarningLevel = 0;
        distractionCriticalTriggered = false;

        if (previous >= 3000 && !criticalTriggered) {
            addAlert("Driver returned attention toward the road.", "normal");
            if (!eyesClosedSince && yawnCount < 2) {
                $("alert-overlay").classList.add("hidden");
            }
        }
        return;
    }

    if (!distractionSince) {
        distractionSince = now;
    }

    const elapsed = now - distractionSince;

    if (elapsed >= 10000 && distractionWarningLevel < 1) {
        distractionWarningLevel = 1;
        if (bookingActive) distractionIncidentCount++;
        addAlert(
            "Distraction warning — head has been away from forward position for about 10 seconds.",
            "caution"
        );
        $("alert-overlay").classList.remove("hidden");
        $("alert-overlay-text").textContent = "⚠  DISTRACTION WARNING — LOOK FORWARD";
        playAlarmTone(780, 180);
    }

    if (elapsed >= 15000 && distractionWarningLevel < 2) {
        distractionWarningLevel = 2;
        addAlert(
            "Second distraction warning — 15 seconds of continuous head diversion.",
            "critical"
        );
        $("alert-overlay").classList.remove("hidden");
        $("alert-overlay-text").textContent = "🚨 SECOND DISTRACTION WARNING";
        playAlarmTone(950, 250);
    }

    if (elapsed >= 16000 && !distractionCriticalTriggered) {
        distractionCriticalTriggered = true;

        addAlert(
            "ALARM — distraction may lead to accident!!",
            "critical"
        );

        showCriticalAlert({
            title: "DISTRACTION ALERT",
            message: "Driver attention has been away from the forward direction for too long.",
            durationLabel: "CONTINUOUS DISTRACTION: " + secondsLabel(elapsed),
            instruction: "DISTRACTION MAY LEAD TO ACCIDENT!! — LOOK FORWARD AND DRIVE SAFELY.",
            speech: "Warning. Distraction may lead to accident. Please look forward and focus on the road."
        });
    }
}

function secondsLabel(ms) {
    return (ms / 1000).toFixed(1) + " SECONDS";
}

function updateEyeClosureTimer(now) {
    if (!eyesClosedSince) {
        closureTime.textContent = "0.0s";
        closureBar.style.width = "0%";
        return;
    }

    const elapsed = now - eyesClosedSince;
    const seconds = elapsed / 1000;

    if (bookingActive) maxEyeClosureMs = Math.max(maxEyeClosureMs, elapsed);
    closureTime.textContent = seconds.toFixed(1) + "s";
    closureBar.style.width =
        Math.min(100, (elapsed / CRITICAL_EYE_CLOSURE_MS) * 100) + "%";

    if (elapsed < 1000) {
        closureTime.className = "mono text-xl font-bold text-emerald-400";
        closureBar.className = "h-full bg-emerald-500 transition-all";
    } else if (elapsed < 3000) {
        closureTime.className = "mono text-xl font-bold text-amber-400";
        closureBar.className = "h-full bg-amber-500 transition-all";
    } else if (elapsed < 5000) {
        closureTime.className = "mono text-xl font-bold text-orange-400";
        closureBar.className = "h-full bg-orange-500 transition-all";
    } else {
        closureTime.className = "mono text-xl font-bold text-red-400 pulse-alert";
        closureBar.className = "h-full bg-red-500 transition-all";
    }

    if (elapsed >= CRITICAL_EYE_CLOSURE_MS) {
        criticalDuration.textContent = (elapsed / 1000).toFixed(1);
        showCriticalAlert({
            title: "CRITICAL DRIVER ALERT",
            message: "Prolonged eye closure detected.",
            durationLabel: "EYES CLOSED: " + (elapsed / 1000).toFixed(1) + " SECONDS",
            instruction: "WAKE UP AND STOP SAFELY AS SOON AS POSSIBLE",
            speech: "Critical driver alert. Prolonged eye closure detected. Wake up and stop safely as soon as possible."
        });
    }
}

function handleEyeState(eyesClosed, now) {
    if (eyesClosed) {
        if (!eyesClosedSince) {
            eyesClosedSince = now;
            warningLevel = 0;
        }

        const elapsed = now - eyesClosedSince;

        if (elapsed >= 5000 && warningLevel < 3) {
            warningLevel = 3;
            addAlert("Critical warning — eyes closed for more than 5 seconds.", "critical");
            $("alert-overlay").classList.remove("hidden");
            $("alert-overlay-text").textContent = "🚨 CRITICAL DROWSINESS WARNING";
            playAlarmTone(950, 250);
        } else if (elapsed >= 3000 && warningLevel < 2) {
            warningLevel = 2;
            addAlert("Strong warning — prolonged eye closure detected.", "critical");
            $("alert-overlay").classList.remove("hidden");
            $("alert-overlay-text").textContent = "⚠  PROLONGED EYE CLOSURE";
            playAlarmTone(850, 180);
        } else if (elapsed >= 1000 && warningLevel < 1) {
            warningLevel = 1;
            addAlert("Possible drowsiness — eye closure lasting over 1 second.", "caution");
            $("alert-overlay").classList.remove("hidden");
            $("alert-overlay-text").textContent = "⚠  DROWSINESS WARNING";
            playAlarmTone(700, 140);
        }
    } else {
        const previousDuration = eyesClosedSince ? now - eyesClosedSince : 0;

        // A normal blink is cleared quickly. If a longer closure ends,
        // record it as an event and restore the normal state.
        if (previousDuration >= 1000 && previousDuration < CRITICAL_EYE_CLOSURE_MS) {
            addAlert(
                "Driver eyes reopened after " + (previousDuration / 1000).toFixed(1) + " seconds.",
                "caution"
            );
        }

        eyesClosedSince = null;
        warningLevel = 0;

        if (!criticalTriggered) {
            $("alert-overlay").classList.add("hidden");
        }
    }
}

// ============================================================
// PHASE 1 — UNIFIED RISK ENGINE
// ============================================================

function calculateRiskProfile(ear, mar, headState, now) {

    let score = 100;
    const reasons = [];

    // --------------------------------------------------------
    // 1. EYE CLOSURE / DROWSINESS
    // --------------------------------------------------------

    let eyeClosureSeconds = 0;

    if (eyesClosedSince) {
        eyeClosureSeconds =
            (now - eyesClosedSince) / 1000;
    }

    if (eyeClosureSeconds >= 8) {

        score -= 55;

        reasons.push({
            type: "drowsiness",
            severity: "critical",
            text: "Prolonged eye closure"
        });

    } else if (eyeClosureSeconds >= 5) {

        score -= 38;

        reasons.push({
            type: "drowsiness",
            severity: "high",
            text: "Extended eye closure"
        });

    } else if (eyeClosureSeconds >= 3) {

        score -= 22;

        reasons.push({
            type: "drowsiness",
            severity: "medium",
            text: "Sustained eye closure"
        });

    } else if (eyeClosureSeconds >= 1) {

        score -= 8;

        reasons.push({
            type: "drowsiness",
            severity: "low",
            text: "Possible drowsiness"
        });
    }


    // --------------------------------------------------------
    // 2. CURRENT YAWN
    // --------------------------------------------------------

    if (mar > YAWN_MAR) {

        score -= 4;

        reasons.push({
            type: "yawn",
            severity: "low",
            text: "Yawning detected"
        });
    }


    // --------------------------------------------------------
    // 3. CUMULATIVE YAWNS
    // --------------------------------------------------------

    if (yawnCount >= 5) {

        score -= 22;

        reasons.push({
            type: "fatigue",
            severity: "critical",
            text: "Repeated yawning detected"
        });

    } else if (yawnCount >= 2) {

        score -= 12;

        reasons.push({
            type: "fatigue",
            severity: "medium",
            text: "Multiple yawns detected"
        });

    } else if (yawnCount >= 1) {

        score -= 4;

        reasons.push({
            type: "fatigue",
            severity: "low",
            text: "Yawn detected"
        });
    }


    // --------------------------------------------------------
    // 4. HEAD ORIENTATION
    // --------------------------------------------------------

    if (headState === "Turned Away") {

        score -= 15;

        reasons.push({
            type: "distraction",
            severity: "high",
            text: "Driver looking away from road"
        });

    } else if (headState === "Downward") {

        score -= 12;

        reasons.push({
            type: "distraction",
            severity: "high",
            text: "Driver looking downward"
        });
    }


    // --------------------------------------------------------
    // 5. CONTINUOUS DISTRACTION
    // --------------------------------------------------------

    let distractionSeconds = 0;

    if (distractionSince) {

        distractionSeconds =
            (now - distractionSince) / 1000;

        if (distractionSeconds >= 16) {

            score -= 45;

            reasons.push({
                type: "distraction",
                severity: "critical",
                text: "Prolonged distraction"
            });

        } else if (distractionSeconds >= 15) {

            score -= 30;

            reasons.push({
                type: "distraction",
                severity: "high",
                text: "Extended distraction"
            });

        } else if (distractionSeconds >= 10) {

            score -= 18;

            reasons.push({
                type: "distraction",
                severity: "medium",
                text: "Continuous distraction"
            });

        } else if (distractionSeconds >= 3) {

            score -= 6;

            reasons.push({
                type: "distraction",
                severity: "low",
                text: "Attention drifting"
            });
        }
    }


    // --------------------------------------------------------
    // 6. MULTI-SIGNAL RISK FUSION
    // --------------------------------------------------------
    // When multiple independent signals appear together,
    // increase the risk slightly.

    if (
        eyeClosureSeconds >= 3 &&
        yawnCount >= 2
    ) {

        score -= 8;

        reasons.push({
            type: "combined",
            severity: "high",
            text: "Multiple fatigue signals detected"
        });
    }


    if (
        distractionSeconds >= 10 &&
        eyeClosureSeconds >= 1
    ) {

        score -= 8;

        reasons.push({
            type: "combined",
            severity: "high",
            text: "Distraction and eye-closure signals combined"
        });
    }


    // --------------------------------------------------------
    // 7. FINAL SCORE
    // --------------------------------------------------------

    score = Math.max(
        0,
        Math.min(100, Math.round(score))
    );


    // --------------------------------------------------------
    // 8. RISK LEVEL
    // --------------------------------------------------------

    let level = "SAFE";

    if (score >= 85) {

        level = "SAFE";

    } else if (score >= 70) {

        level = "LOW RISK";

    } else if (score >= 50) {

        level = "HIGH RISK";

    } else {

        level = "CRITICAL";
    }


    return {
        score,
        level,
        reasons
    };
}


// Compatibility wrapper.
// Existing parts of SafeRide can still call calculateSafetyScore().
function calculateSafetyScore(ear, mar, headState, now) {

    const profile =
        calculateRiskProfile(
            ear,
            mar,
            headState,
            now
        );

    currentRiskScore = profile.score;
    currentRiskLevel = profile.level;
    currentRiskReasons = profile.reasons;

    return profile.score;
}


function processFaceResult(result, now) {
    // ------------------------------------------------------------
    // NO FACE
    // ------------------------------------------------------------
    if (
        !result ||
        !result.faceLandmarks ||
        result.faceLandmarks.length === 0
    ) {
        if (!noFaceSince) {
            noFaceSince = now;
        }

        $("face-box").style.display = "none";
        setLiveState("NO FACE", "bg-amber-500");

        setIndicator("ind-eye", "bg-gray-600");
        setIndicator("ind-yawn", "bg-gray-600");
        setIndicator("ind-head", "bg-gray-600");
        setIndicator("ind-distract", "bg-gray-600");

        $("ear-val").textContent = "--";
        $("mar-val").textContent = "--";
        $("head-val").textContent = "No face";

        // MediaPipe can briefly lose a face for a few frames.
        if (now - noFaceSince >= FACE_LOSS_GRACE_MS) {
            eyesClosedSince = null;
            mouthOpenSince = null;
            distractionSince = null;

            stableEyesClosed = false;
            headCandidateState = "Stable";
            headCandidateSince = null;
            stableHeadState = "Stable";

            smoothedEAR = null;
            smoothedMAR = null;

            currentRiskLevel = "UNKNOWN";
            currentRiskScore = 0;
            currentRiskReasons = [];

            $("score-text").textContent = "--";
            $("score-status").textContent = "NO FACE";
        }

        updateRiskUI("UNKNOWN", 0, []);
        updateEyeClosureTimer(now);
        updateDistractionUI(now);
        return;
    }

    // Face is visible again.
    noFaceSince = null;

    const lm = result.faceLandmarks[0];

    updateFaceBox(lm);
    drawLandmarks(lm);

    // ------------------------------------------------------------
    // RAW SIGNALS
    // ------------------------------------------------------------
    const leftEAR = eyeAspectRatio(lm, LEFT_EYE);
    const rightEAR = eyeAspectRatio(lm, RIGHT_EYE);

    const rawEAR = (leftEAR + rightEAR) / 2;
    const rawMAR = mouthAspectRatio(lm);
    const rawHead = estimateHeadState(lm);

    // ------------------------------------------------------------
    // SIGNAL SMOOTHING
    // ------------------------------------------------------------
    smoothedEAR = smoothSignal(smoothedEAR, rawEAR);
    smoothedMAR = smoothSignal(smoothedMAR, rawMAR);

    const ear = smoothedEAR ?? rawEAR;
    const mar = smoothedMAR ?? rawMAR;

    // ------------------------------------------------------------
    // STABLE STATES
    // ------------------------------------------------------------
    const eyesClosed = getStableEyeState(ear);

    const stableHead = getStableHeadState(
        rawHead.label,
        now
    );

    const head = {
        label: stableHead,
        score: stableHead === "Stable" ? 0 : 1
    };

    // ------------------------------------------------------------
    // YAWN EVENT DETECTION
    // ------------------------------------------------------------
    const yawnDetected = mar > YAWN_MAR;

    if (yawnDetected) {
        if (!mouthOpenSince) {
            mouthOpenSince = now;
            yawnInProgress = false;
        }

        if (
            !yawnInProgress &&
            now - mouthOpenSince >= YAWN_HOLD_MS
        ) {
            yawnInProgress = true;
        }
    } else {
        // Count once when the sustained mouth-open period ends.
        if (yawnInProgress) {
            registerYawn(now);
        }

        mouthOpenSince = null;
        yawnInProgress = false;
    }

    // ------------------------------------------------------------
    // EXISTING INTERVENTION LOGIC
    // ------------------------------------------------------------
    handleEyeState(eyesClosed, now);
    updateEyeClosureTimer(now);

    handleDistraction(head.label, now);
    updateDistractionUI(now);

    // ------------------------------------------------------------
    // UNIFIED RISK ENGINE
    // ------------------------------------------------------------
    const riskProfile = calculateRiskProfile(
        ear,
        mar,
        head.label,
        now
    );

    currentRiskScore = riskProfile.score;
    currentRiskLevel = riskProfile.level;
    currentRiskReasons = riskProfile.reasons;

    // ------------------------------------------------------------
    // FINAL SCORE SMOOTHING
    // ------------------------------------------------------------
    alertness =
        alertness * 0.72 +
        riskProfile.score * 0.28;

    alertness = Math.max(
        0,
        Math.min(100, alertness)
    );

    updateGauge(alertness);

    // Risk Engine panel shows the current risk profile.
    updateRiskUI(
        riskProfile.level,
        riskProfile.score,
        riskProfile.reasons
    );

    // ------------------------------------------------------------
    // LIVE DASHBOARD
    // ------------------------------------------------------------
    updateMetrics(
        ear,
        mar,
        head.label
    );

    setIndicator(
        "ind-eye",
        eyesClosed ? "bg-red-500" : "bg-emerald-500",
        eyesClosed
    );

    setIndicator(
        "ind-yawn",
        yawnDetected ? "bg-amber-500" : "bg-emerald-500",
        yawnDetected
    );

    setIndicator(
        "ind-head",
        head.score
            ? (
                head.label === "Turned Away"
                    ? "bg-amber-500"
                    : "bg-orange-500"
            )
            : "bg-emerald-500",
        !!head.score
    );

    if (head.score) {
        $("ind-distract").className =
            "w-3 h-3 rounded-full bg-amber-500 pulse-alert";
    } else {
        $("ind-distract").className =
            "w-3 h-3 rounded-full bg-emerald-500";
    }

    setLiveState(
        "FACE TRACKING",
        "bg-emerald-500"
    );
}

async function initFaceLandmarker() {
    setModelStatus("AI LOADING", "loading");

    try {
        const vision = await FilesetResolver.forVisionTasks(MP_WASM);

        faceLandmarker = await FaceLandmarker.createFromOptions(
            vision,
            {
                baseOptions: {
                    modelAssetPath: FACE_MODEL,
                    delegate: "CPU"
                },
                runningMode: "VIDEO",
                numFaces: 1,
                minFaceDetectionConfidence: 0.5,
                minFacePresenceConfidence: 0.5,
                minTrackingConfidence: 0.5,
                outputFaceBlendshapes: false,
                outputFacialTransformationMatrixes: false
            }
        );

        setModelStatus("AI READY", "ready");
    } catch (error) {
        console.error(error);
        setModelStatus("AI ERROR", "error");
        camMsg.textContent =
            "AI model failed to load. Check your internet connection.";
        addAlert("MediaPipe AI model could not be loaded.", "critical");
    }
}

async function startCamera() {
    console.log("START CAMERA: function called");

    if (!navigator.mediaDevices?.getUserMedia) {
        console.error("START CAMERA: getUserMedia unavailable");
        camMsg.textContent = "Camera API unavailable. Use Chrome/Edge on localhost.";
        return false;
    }

    try {
        console.log("START CAMERA: requesting camera permission...");

        // Clean up any previous stream
        if (cameraStream) {
            console.warn("🛑 CAMERA STOP: startCamera old-stream cleanup");
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }

        // Use a lighter resolution for MediaPipe + browser stability
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: "user" },
                width: { ideal: 640, max: 1280 },
                height: { ideal: 480, max: 720 },
                frameRate: { ideal: 24, max: 30 }
            },
            audio: false
        });

        console.log("START CAMERA: permission granted");
        console.log("========== CAMERA CHECKPOINT ==========");

        console.log("STREAM:", stream);
        console.log("TRACKS:", stream.getTracks());

        const testTrack = stream.getVideoTracks()[0];

        if (testTrack) {
            console.log("TRACK READY STATE:", testTrack.readyState);
            console.log("TRACK SETTINGS:", testTrack.getSettings());
            console.log("TRACK ENABLED:", testTrack.enabled);
        }

        console.log("VIDEO BEFORE:", {
            readyState: video.readyState,
            videoWidth: video.videoWidth,
            videoHeight: video.videoHeight,
            paused: video.paused
        });

        video.srcObject = stream;
        video.muted = true;
        video.autoplay = true;
        video.playsInline = true;

        await video.play();

        console.log("========== CAMERA CHECKPOINT AFTER PLAY ==========");

        console.log("VIDEO AFTER:", {
            readyState: video.readyState,
            videoWidth: video.videoWidth,
            videoHeight: video.videoHeight,
            paused: video.paused
        });

        console.log("TRACK AFTER:", {
            readyState: testTrack?.readyState,
            enabled: testTrack?.enabled
        });

        console.log("CAMERA CHECKPOINT SUCCESS");

        const tracks = stream.getVideoTracks();

        if (!tracks.length) {
            throw new Error("No video track returned by camera.");
        }

        const track = tracks[0];

        console.log("CAMERA TRACK:", track);
        console.log("CAMERA SETTINGS:", track.getSettings());

        cameraStream = stream;

        // IMPORTANT: configure video BEFORE attaching stream
        video.muted = true;
        video.autoplay = true;
        video.playsInline = true;
        video.setAttribute("autoplay", "");
        video.setAttribute("muted", "");
        video.setAttribute("playsinline", "");

        video.srcObject = stream;

        console.log("START CAMERA: stream attached");

        // Detect if browser pauses the video
        video.onpause = async () => {
            if (bookingActive && cameraStream && !video.ended) {
                console.warn("CAMERA WATCHDOG: video paused. Restarting...");
                try {
                    await video.play();
                    console.log("CAMERA WATCHDOG: video restarted");
                } catch (e) {
                    console.error("CAMERA WATCHDOG: restart failed", e);
                }
            }
        };

        video.onended = () => {
            console.error("CAMERA WATCHDOG: video ended unexpectedly");

            if (bookingActive) {
                camMsg.textContent = "Camera stopped unexpectedly. Attempting recovery...";
                recoverCamera();
            }
        };

        // Detect camera track stopping
        track.onended = () => {
            console.error("CAMERA TRACK ENDED");

            if (bookingActive) {
                camMsg.textContent = "Camera disconnected. Attempting recovery...";
                recoverCamera();
            }
        };

        track.onmute = () => {
            console.warn("CAMERA TRACK MUTED");
        };

        track.onunmute = () => {
            console.log("CAMERA TRACK UNMUTED");
        };

        // Wait for metadata
        await new Promise((resolve, reject) => {
            if (video.readyState >= 1) {
                resolve();
                return;
            }

            const timeout = setTimeout(() => {
                reject(new Error("Camera metadata timeout."));
            }, 5000);

            video.addEventListener("loadedmetadata", () => {
                clearTimeout(timeout);
                resolve();
            }, { once: true });
        });

        console.log(
            "CAMERA METADATA:",
            video.videoWidth,
            "x",
            video.videoHeight
        );

        await video.play();

        console.log(
            "START CAMERA: video.play() successful",
            video.videoWidth,
            "x",
            video.videoHeight
        );

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        camFallback.classList.add("hidden");
        overlayMetrics.classList.remove("hidden");
        liveStatus.classList.remove("hidden");

        setLiveState("CAMERA ACTIVE", "bg-emerald-500");

        console.log("START CAMERA: CAMERA ACTIVE");

        return true;

    } catch (error) {
        console.error("START CAMERA ERROR:", error);

        if (cameraStream) {
            console.warn("🛑 CAMERA STOP: startCamera error cleanup");
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }

        video.srcObject = null;

        camFallback.classList.remove("hidden");

        camMsg.textContent =
            "Camera could not be started: " + error.message;

        addAlert(
            "Camera could not be started: " + error.message,
            "critical"
        );

        return false;
    }
}
let cameraRecovering = false;

async function recoverCamera() {
    if (cameraRecovering || !bookingActive) return;

    cameraRecovering = true;

    console.warn("CAMERA RECOVERY: restarting camera...");

    try {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }

        video.srcObject = null;

        await new Promise(resolve => setTimeout(resolve, 500));

        const recovered = await startCamera();

        if (recovered) {
            console.log("CAMERA RECOVERY: SUCCESS");

            camMsg.textContent = "Camera monitoring restored.";
            setLiveState("CAMERA ACTIVE", "bg-emerald-500");

            lastVideoTime = -1;
            frameCounter = 0;

            scheduleNextFrame();
        } else {
            console.error("CAMERA RECOVERY: FAILED");
        }

    } catch (error) {
        console.error("CAMERA RECOVERY ERROR:", error);
    } finally {
        cameraRecovering = false;
    }
}



// CAMERA WATCHDOG
function startCameraWatchdog() {
    clearInterval(cameraWatchdog);

    cameraWatchdog = setInterval(async () => {

        // Do nothing when trip is not active
        if (!bookingActive || !cameraStream) {
            return;
        }

        const track = cameraStream.getVideoTracks()[0];

        // Camera track disappeared
        if (!track) {
            console.warn("WATCHDOG: No camera track");
            recoverCamera();
            return;
        }

        // Camera track stopped
        if (track.readyState !== "live") {
            console.warn("WATCHDOG: Camera track is not live");
            recoverCamera();
            return;
        }

        // Video unexpectedly paused
        if (video.paused && !video.ended) {
            console.warn("WATCHDOG: Video paused");

            try {
                await video.play();
                console.log("WATCHDOG: Video restarted");
            } catch (error) {
                console.error("WATCHDOG: Video restart failed", error);
                recoverCamera();
            }
        }

    }, 2000);
}

function stopCameraWatchdog() {
    if (cameraWatchdog) {
        clearInterval(cameraWatchdog);
        cameraWatchdog = null;
        console.log("CAMERA WATCHDOG: stopped");
    }
}

function stopCamera() {
    if (cameraStream) {
        console.warn("🛑 CAMERA STOP: stopCamera()");
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    $("face-box").style.display = "none";
    camFallback.classList.remove("hidden");

    overlayMetrics.classList.add("hidden");
    liveStatus.classList.add("hidden");

    camMsg.textContent = "Camera is inactive";
}

function processVideoFrame() {
    if (!bookingActive || !faceLandmarker) {
        return;
    }

    // Make sure the camera stream still exists
    if (!cameraStream) {
        scheduleNextFrame();
        return;
    }

    // Check the camera track
    const track = cameraStream.getVideoTracks()[0];

    if (!track || track.readyState !== "live") {
        console.warn("VIDEO FRAME: camera track is not live");
        //recoverCamera();
        return;
    }

    // Wait until the video has usable dimensions
    if (video.readyState < 2 || !video.videoWidth) {
        scheduleNextFrame();
        return;
    }

    const now = performance.now();

    if (video.currentTime === lastVideoTime) {
        scheduleNextFrame();
        return;
    }

    lastVideoTime = video.currentTime;
    frameCounter++;

    // Process every 3rd frame to reduce CPU load
    if (frameCounter % 3 !== 0) {
        updateEyeClosureTimer(now);
        scheduleNextFrame();
        return;
    }

    if (processing) {
        scheduleNextFrame();
        return;
    }

    processing = true;

    try {
        const result = faceLandmarker.detectForVideo(video, now);
        //const result = null;

        inferenceCounter++;

        processFaceResult(result, now);

    } catch (error) {
        console.error("Face detection error:", error);

    } finally {
        processing = false;
    }

    if (now - lastFpsTime >= 1000) {
        aiFps =
            inferenceCounter * 1000 /
            (now - lastFpsTime);

        inferenceCounter = 0;
        lastFpsTime = now;
    }

    scheduleNextFrame();
}

function scheduleNextFrame() {
    if (!bookingActive) return;

    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
        video.requestVideoFrameCallback(() => processVideoFrame());
    } else {
        animationId = requestAnimationFrame(processVideoFrame);
    }
}

async function startBackendTrip() {
    try {
        const response = await fetch(`${API_BASE_URL}/trips/start`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                driver_id: DRIVER_ID
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to start backend trip.");
        }

        console.log("Backend trip started:", data);

        return data.trip.trip_id;

    } catch (error) {
        console.error("Backend trip start error:", error);

        addAlert(
            "Backend connection failed. Trip was not started.",
            "critical"
        );

        return null;
    }
}
async function endBackendTrip() {
    const tripId = window.safeRideTripId;

    if (!tripId) {
        console.warn("No backend trip ID found. Nothing to end.");
        return true;
    }

    try {
        
    // Save final safety metrics before ending the trip.
    const safetyResponse = await fetch(
        `${API_BASE_URL}/trips/${tripId}/safety-data`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                eye_closure_duration: maxEyeClosureMs / 1000,
                yawn_count: yawnCount,
                distraction_duration:
                    distractionIncidentCount > 0 ? 1 : 0,
                safety_score: calculateTripRating() * 10,
                alertness_score: alertness
            })
        }
    );

    if (!safetyResponse.ok) {
        throw new Error("Could not save trip safety metrics.");
    }

        
        const response = await fetch(
            `${API_BASE_URL}/trips/${tripId}/end`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                }
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Unable to end backend trip."
            );
        }

        
        // Sync the visible report with the backend's saved score.
        const finalScore = Number(data.trip_report?.final_score);

        if (Number.isFinite(finalScore) && finalScore >= 0 && finalScore <= 10) {
            const stars = Math.round(finalScore / 2);

            $("report-score").textContent =
                finalScore.toFixed(1) + " / 10";
            $("report-stars").textContent = starsForScore(finalScore);
            $("report-rating-text").textContent = ratingText(finalScore);

            const progress = finalScore * 10;
            $("report-progress").style.width = progress + "%";
            $("report-progress-label").textContent =
                Math.round(progress) + "%";

            // Update the locally saved report to match the backend.
            try {
                const saved = JSON.parse(
                    localStorage.getItem("saferide_last_trip_report") || "null"
                );

                if (saved) {
                    saved.score = Number(finalScore.toFixed(1));
                    saved.stars = stars;
                    localStorage.setItem(
                        "saferide_last_trip_report",
                        JSON.stringify(saved)
                    );
                    lastSavedTrip = saved;
                }
            } catch (storageError) {
                console.warn("Could not sync local trip report:", storageError);
            }
        }


        console.log("Backend trip ended:", data);

        window.safeRideTripId = null;

        return true;

    } catch (error) {
        console.error("Backend trip end error:", error);

        addAlert(
            "Backend could not end the trip: " + error.message,
            "critical"
        );

        return false;
    }
}

async function startMonitoring() {
    if (tripStarting || bookingActive) {
        return;
    }

    if (!faceLandmarker) {
        addAlert(
            "AI is still loading. Please wait a moment and try again.",
            "caution"
        );
        return;
    }

    tripStarting = true;
    bookingBtn.disabled = true;
    bookingBtn.textContent = "Starting...";

    let backendTripId = null;

    try {
        // 1. Start camera first.
        const cameraStarted = await startCamera();

        if (!cameraStarted) {
            addAlert(
                "Camera could not be started. Trip was not started.",
                "critical"
            );
            return;
        }

        // 2. Create the backend trip only after the camera is ready.
        backendTripId = await startBackendTrip();

        if (!backendTripId) {
            stopCamera();
            addAlert(
                "Backend trip could not be started. Camera stopped.",
                "critical"
            );
            return;
        }

        window.safeRideTripId = backendTripId;

        // 3. Mark the trip ACTIVE immediately.
        // Dashboard rendering below must never be allowed to kill
        // an already-running camera/backend session.
        bookingActive = true;

        startCameraWatchdog();

        bookingBtn.textContent = "End Trip";
        bookingBtn.classList.remove(
            "bg-emerald-400",
            "hover:bg-emerald-300"
        );
        bookingBtn.classList.add(
            "bg-red-400",
            "hover:bg-red-300"
        );

        modeBadge.textContent = "MONITORING";
        modeBadge.className =
            "px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-400";

        // 4. Initialize dashboard values separately.
        try {
            const restrictionStatus = $("restriction-status");
            if (restrictionStatus) {
                restrictionStatus.textContent =
                    "Safety Mode active — distracting apps simulated as restricted.";
            }

            document
                .querySelectorAll('[data-app="blocked"]')
                .forEach(el => el.classList.add("disabled-card"));

            alertness = 98;
            tripStartTime = performance.now();
            tripElapsedMs = 0;
            maxEyeClosureMs = 0;
            tripWarningCount = 0;
            distractionIncidentCount = 0;
            tripCriticalCount = 0;
            tripEventCount = 0;

            const reportState = $("report-state");
            if (reportState) {
                reportState.textContent =
                    "Trip active • collecting safety performance";
            }

            const reportStorage = $("report-storage");
            if (reportStorage) {
                reportStorage.textContent = "COLLECTING";
                reportStorage.className =
                    "text-[10px] px-2 py-1 rounded-full bg-amber-500/10 text-amber-400";
            }

            updateTripReport(false);
            startTripReportTimer();

            eyesClosedSince = null;
            criticalTriggered = false;
            warningLevel = 0;
            // ========================================================
            // RESET PHASE 1 RISK ENGINE
            // ========================================================

            smoothedEAR = null;
            smoothedMAR = null;

            stableEyesClosed = false;

            stableHeadState = "Stable";
            headCandidateState = "Stable";
            headCandidateSince = null;

            noFaceSince = null;

            currentRiskLevel = "SAFE";
            currentRiskScore = 98;
            currentRiskReasons = [];
           
            updateRiskUI(
            "SAFE",
            98,
            []
            );

            mouthOpenSince = null;
            yawnInProgress = false;
            yawnCount = 0;
            yawnWarningLevel = 0;
            refreshRecommendationShown = false;

            distractionSince = null;
            distractionWarningLevel = 0;
            distractionCriticalTriggered = false;

            const refreshAlert = $("refresh-alert");
            if (refreshAlert) {
                refreshAlert.classList.add("hidden");
            }

            updateYawnUI();
            updateDistractionUI(performance.now());

        } catch (uiError) {
            console.warn(
                "Dashboard initialization warning:",
                uiError
            );
        }

        lastVideoTime = -1;
        frameCounter = 0;

        addAlert(
            "Safety Mode activated. Camera monitoring started.",
            "normal"
        );

        // 5. Start AI processing.
        scheduleNextFrame();

        console.log("SAFERIDE: FULL TRIP ACTIVE", {
            tripId: window.safeRideTripId,
            videoWidth: video.videoWidth,
            videoHeight: video.videoHeight,
            cameraTracks: cameraStream?.getVideoTracks()?.length || 0
        });

    } catch (error) {
        console.error("Unexpected trip start error:", error);

        if (window.safeRideTripId || backendTripId) {
            window.safeRideTripId = window.safeRideTripId || backendTripId;
            await endBackendTrip();
        }

        stopCamera();
        bookingActive = false;
        window.safeRideTripId = null;

        try {
            addAlert(
                "Trip could not be started: " + error.message,
                "critical"
            );
        } catch (alertError) {
            console.error("Could not display startup error:", alertError);
        }

        camMsg.textContent =
            "Trip startup error: " + error.message;

    } finally {
        tripStarting = false;
        bookingBtn.disabled = false;

        if (!bookingActive) {
            bookingBtn.textContent = "Start Trip";
        }
    }
}
   async function stopMonitoring() {
    // Stop camera watchdog
    if (cameraWatchdog) {
        clearInterval(cameraWatchdog);
        cameraWatchdog = null;
    }

    // Save the current trip report before resetting state
    if (tripStartTime) {
        tripElapsedMs = performance.now() - tripStartTime;
        updateTripReport(true);
    }

    // Remember the backend trip ID before clearing it
    const backendTripId = window.safeRideTripId;

    // -------------------------------------------------
    // 1. STOP FRONTEND MONITORING
    // -------------------------------------------------
    stopTripReportTimer();

    tripStartTime = null;
    bookingActive = false;

    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }

    hideCriticalAlert();

    // -------------------------------------------------
    // 2. STOP CAMERA
    // -------------------------------------------------
    stopCameraWatchdog();
    stopCamera();

    // -------------------------------------------------
    // 3. END BACKEND TRIP
    // -------------------------------------------------
    if (backendTripId) {
        await endBackendTrip();
    }

    // -------------------------------------------------
    // 4. RESET BUTTON/UI
    // -------------------------------------------------
    bookingBtn.disabled = false;
    bookingBtn.textContent = "Start Trip";

    bookingBtn.classList.remove(
        "bg-red-400",
        "hover:bg-red-300"
    );

    bookingBtn.classList.add(
        "bg-emerald-400",
        "hover:bg-emerald-300"
    );

    modeBadge.textContent = "INACTIVE";

    modeBadge.className =
        "px-3 py-1 rounded-full text-xs font-medium bg-gray-800 text-gray-400";

    $("restriction-status").textContent =
        "Safety Mode inactive.";

    document
        .querySelectorAll('[data-app="blocked"]')
        .forEach(el => el.classList.remove("disabled-card"));

    $("alert-overlay").classList.add("hidden");

    // -------------------------------------------------
    // 5. RESET AI STATE
    // -------------------------------------------------
    eyesClosedSince = null;
    warningLevel = 0;

    mouthOpenSince = null;
    yawnInProgress = false;
    yawnCount = 0;
    yawnWarningLevel = 0;
    refreshRecommendationShown = false;

    distractionSince = null;
    distractionWarningLevel = 0;
    distractionCriticalTriggered = false;

    $("refresh-alert").classList.add("hidden");

    // -------------------------------------------------
    // 6. RESET DASHBOARD VALUES
    // -------------------------------------------------
    closureTime.textContent = "0.0s";
    closureBar.style.width = "0%";

    updateYawnUI();
    updateDistractionUI(performance.now());

    $("ear-val").textContent = "--";
    $("mar-val").textContent = "--";
    $("head-val").textContent = "--";
    $("fps-val").textContent = "--";

    setIndicator("ind-eye", "bg-emerald-500");
    setIndicator("ind-yawn", "bg-emerald-500");
    setIndicator("ind-head", "bg-emerald-500");
    setIndicator("ind-distract", "bg-emerald-500");

    alertness = 98;

    smoothedEAR = null;
    smoothedMAR = null;
    stableEyesClosed = false;
    stableHeadState = "Stable";
    headCandidateState = "Stable";
    headCandidateSince = null;
    noFaceSince = null;

    currentRiskLevel = "SAFE";
    currentRiskScore = 98;
    currentRiskReasons = [];

    updateGauge(98);
    updateRiskUI("SAFE", 98, []);

    // Make absolutely sure the frontend no longer
    // remembers the completed backend trip.
    window.safeRideTripId = null;

    addAlert(
        backendTripId
            ? "Trip ended successfully. Safety Mode deactivated."
            : "Trip ended. Safety Mode deactivated.",
        "normal"
    );

    console.log(
        "SafeRide trip ended:",
        backendTripId || "No backend trip"
    );
}

// Prevent accidental form submission/page reload if the button
// is inside a <form>. SafeRide must remain a single-page session.
if (bookingBtn) {
    bookingBtn.type = "button";
    bookingBtn.addEventListener("click", (event) => {
        event.preventDefault();
    });
}

window.toggleBooking = async function() {

    if (tripStarting) {
        return;
    }

    if (bookingActive) {
        await stopMonitoring();
    } else {
        await startMonitoring();
    }

};

window.addEventListener("resize", () => {
    if (video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }
});

window.addEventListener("beforeunload", () => {
    console.warn("⚠️ PAGE UNLOADING");

    stopCriticalAlarm();

    // Do not manually stop the camera here.
    // The browser will release the camera when the page is actually closed.
});

if (window.lucide) {
    lucide.createIcons();
}

loadLastTripReport();
initFaceLandmarker();

const BACKEND_URL = "http://127.0.0.1:8000";

const authScreen = document.getElementById("auth-screen");
const googleLoginBtn = document.getElementById("google-login-btn");

async function checkAuthentication() {
    try {
        const response = await fetch(`${BACKEND_URL}/auth/me`, {
            credentials: "include"
        });

        if (!response.ok) {
            showLoginScreen();
            return;
        }

        const data = await response.json();

        if (data.authenticated) {
            showApp(data.user);
        } else {
            showLoginScreen();
        }

    } catch (error) {
        console.error("Authentication check failed:", error);
        showLoginScreen();
    }
}

function showLoginScreen() {
    if (authScreen) {
        authScreen.style.display = "flex";
    }
}

function showApp(user) {
    if (authScreen) {
        authScreen.style.display = "none";
    }

    console.log("Logged-in SafeRide driver:", user);

    // Make the authenticated driver available
    // to the rest of the frontend.
    window.safeRideUser = user;
    DRIVER_ID = user.driver_id;
    loadTripHistory();

    // Populate Driver Profile
    const profileName = document.getElementById("profile-name");
    const profileEmail = document.getElementById("profile-email");
    const profileDriverId = document.getElementById("profile-driver-id");
    const profilePhone = document.getElementById("profile-phone");
    const profileVehicle = document.getElementById("profile-vehicle");
    const profileTrips = document.getElementById("profile-trips");
    const profileAverageScore =
        document.getElementById("profile-average-score");

    if (profileName) {
        profileName.textContent =
            user.name || "SafeRide Driver";
    }

    if (profileEmail) {
        profileEmail.textContent =
            user.email || "Email unavailable";
    }

    if (profileDriverId) {
        profileDriverId.textContent =
            user.driver_id || "--";
    }

    if (profilePhone) {
        profilePhone.textContent =
            user.phone || "Not added";
    }

    if (profileVehicle) {
        profileVehicle.textContent =
            user.vehicle_number || "Not added";
    }

    if (profileTrips) {
        profileTrips.textContent =
            user.total_trips ?? 0;
    }

   
        
    if (profileAverageScore) {
        const score = Number(user.average_score || 0);

        profileAverageScore.textContent =
            score > 0
                ? `${score.toFixed(1)} / 10`
                : "Not rated yet";
    }

    
}

checkAuthentication();

// ============================================================
// SAFE RIDE AI - LOGOUT
// ============================================================

const logoutBtn = document.getElementById("logout-btn");

if (logoutBtn) {
    logoutBtn.addEventListener("click", async () => {

        try {
            const response = await fetch(
                `${BACKEND_URL}/auth/logout`,
                {
                    method: "GET",
                    credentials: "include"
                }
            );

            if (!response.ok) {
                throw new Error("Logout failed.");
            }

            window.safeRideUser = null;
            DRIVER_ID = null;

            showLoginScreen();

        } catch (error) {
            console.error("Logout error:", error);
        }
    });
}

// ============================================================
// SAFE RIDE AI - TRIP HISTORY
// ============================================================

async function loadTripHistory() {

    if (!DRIVER_ID) {
        console.warn("Trip history skipped: no authenticated driver.");
        return;
    }

    const historyCount =
        document.getElementById("history-count");

    const historyList =
        document.getElementById("trip-history-list");

    if (!historyList) {
        return;
    }

    try {

        const response = await fetch(
            `${API_BASE_URL}/drivers/${DRIVER_ID}/history`
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Unable to load trip history."
            );
        }

        const trips = data.trips || [];
        const insightsPanel = document.getElementById("driver-insights-panel");
        if (insightsPanel) {
            insightsPanel.classList.remove("hidden");
            await loadDriverInsights();
        }

        if (historyCount) {
            historyCount.textContent = trips.length;
        }

        if (trips.length === 0) {

            historyList.innerHTML = `
                <div id="trip-history-empty"
                     class="rounded-xl border border-dashed border-gray-800 p-8 text-center">

                    <div class="text-3xl mb-2">🛣️</div>

                    <p class="text-sm text-gray-400">
                        No completed trips yet.
                    </p>

                    <p class="text-xs text-gray-600 mt-1">
                        Complete a SafeRide trip to see it here.
                    </p>

                </div>
            `;

            return;
        }

        historyList.innerHTML = trips
            .map(trip => {

                const startDate = trip.start_time
                    ? new Date(trip.start_time)
                    : null;

                const dateText = startDate
                    ? startDate.toLocaleString()
                    : "Unknown date";

                const score =
                    trip.final_score !== null &&
                    trip.final_score !== undefined
                        ? (Number(trip.final_score) * 10).toFixed(0)
                        : "--";

                const rating =
                    trip.rating !== null &&
                    trip.rating !== undefined
                        ? `${trip.rating}/5`
                        : "--";

                const performance =
                    trip.performance || "Not rated";

                return `
                    <div class="trip-history-item">

                        <div class="trip-history-top">

                            <div>
                                <div class="trip-history-id">
                                    ${trip.trip_id}
                                </div>

                                <div class="trip-history-date">
                                    ${dateText}
                                </div>
                            </div>

                            <span class="trip-history-status">
                                ${(trip.status || "UNKNOWN").toUpperCase()}
                            </span>

                        </div>

                        <div class="trip-history-stats">

                            <div class="trip-history-stat">
                                <div class="trip-history-stat-label">
                                    SAFETY SCORE
                                </div>

                                <div class="trip-history-stat-value">
                                    ${score}/100
                                </div>
                            </div>

                            <div class="trip-history-stat">
                                <div class="trip-history-stat-label">
                                    RATING
                                </div>

                                <div class="trip-history-stat-value">
                                    ${rating}
                                </div>
                            </div>

                            <div class="trip-history-stat">
                                <div class="trip-history-stat-label">
                                    PERFORMANCE
                                </div>

                                <div class="trip-history-stat-value trip-history-performance">
                                    ${performance}
                                </div>
                            </div>

                        </div>

                    </div>
                `;
            })
            .join("");

    } catch (error) {

        console.error(
            "Trip history loading error:",
            error
        );

        if (historyList) {
            historyList.innerHTML = `
                <div class="rounded-xl border border-red-500/20 bg-red-500/5 p-5 text-center">

                    <p class="text-sm text-red-400">
                        Unable to load trip history.
                    </p>

                    <p class="text-xs text-gray-600 mt-1">
                        Please try refreshing the dashboard.
                    </p>

                </div>
            `;
        }
    }
}

// SafeRide AI — Profile modal controls
(() => {
    const modal = document.getElementById("profile-modal");
    const openBtn = document.getElementById("profile-btn");
    const closeBtn = document.getElementById("profile-close-btn");
    const backdrop = document.getElementById("profile-backdrop");

    const historyBtn = document.getElementById("profile-history-btn");
    const reportsBtn = document.getElementById("profile-reports-btn");
    const backBtn = document.getElementById("profile-back-btn");
    const subview = document.getElementById("profile-subview");
    const subviewTitle = document.getElementById("profile-subview-title");
    const subviewContent = document.getElementById("profile-subview-content");
    const actions = document.querySelector(".profile-modal-actions");

    if (!modal || !openBtn) {
        console.error("Profile modal elements are missing from the HTML.");
        return;
    }

    function openProfile() {
        modal.classList.remove("hidden");
        document.body.style.overflow = "hidden";

        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    function closeProfile() {
        modal.classList.add("hidden");
        document.body.style.overflow = "";
        showProfileHome();
    }

    function showProfileHome() {
        if (subview) subview.classList.add("hidden");
        if (actions) actions.classList.remove("hidden");
    }

    
function showSubview(title, content) {
    const insightsPanel = document.getElementById("driver-insights-panel");

    if (insightsPanel) insightsPanel.classList.add("hidden");
    if (subviewTitle) subviewTitle.textContent = title;
    if (subviewContent) subviewContent.innerHTML = content;
    if (actions) actions.classList.add("hidden");
    if (subview) subview.classList.remove("hidden");
}


    openBtn.addEventListener("click", openProfile);
    if (closeBtn) closeBtn.addEventListener("click", closeProfile);
    if (backdrop) backdrop.addEventListener("click", closeProfile);
    if (backBtn) backBtn.addEventListener("click", showProfileHome);

    document.addEventListener("keydown", event => {
        if (event.key === "Escape" && !modal.classList.contains("hidden")) {
            closeProfile();
        }
    });

    if (historyBtn) {
    historyBtn.addEventListener("click", async () => {
        const driverId =
            window.safeRideUser?.driver_id || DRIVER_ID;

        showSubview(
            "Trip History",
            '<p class="text-sm text-gray-400">Loading your trips...</p>'
        );

        if (!driverId) {
            subviewContent.textContent =
                "Driver account could not be identified. Please sign in again.";
            return;
        }

        try {
            const response = await fetch(
                `${BACKEND_URL}/drivers/${encodeURIComponent(driverId)}/history`
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.detail || "Unable to load trip history."
                );
            }

            const trips = data.trips || [];
            
            const insightsPanel = document.getElementById("driver-insights-panel");
                if (insightsPanel) {
                    insightsPanel.classList.remove("hidden");
                    await loadDriverInsights();
                }


            if (trips.length === 0) {
                subviewContent.textContent =
                    "No trips were found for this driver in the database.";
                return;
            }

            subviewContent.replaceChildren();

            trips.forEach((trip) => {
                const card = document.createElement("div");
                card.className = "trip-history-item";

                const id = document.createElement("div");
                id.className = "trip-history-id";
                id.textContent = trip.trip_id || "Trip";

                const date = document.createElement("div");
                date.className = "trip-history-date";
                date.textContent = trip.start_time
                    ? new Date(trip.start_time).toLocaleString()
                    : "Date unavailable";

                const status = document.createElement("div");
                status.className = "trip-history-date";
                status.textContent =
                    `Status: ${trip.status || "Unknown"}`;

                const stats = document.createElement("div");
                stats.className = "trip-history-stats";

                const score = document.createElement("div");
                score.className = "trip-history-stat-value";
                score.textContent =
                    trip.final_score != null
                        ? `Safety score: ${(Number(trip.final_score) * 10).toFixed(0)}/100`
                        : "Safety score: Not available";

                const rating = document.createElement("div");
                rating.className = "trip-history-stat-value";
                rating.textContent =
                    trip.final_score !== null && trip.final_score !== undefined
                        ? `Rating: ${(Number(trip.final_score) / 2).toFixed(1)}/5`
                        : "Rating: Not available";

                const performance = document.createElement("div");
                performance.className = "trip-history-stat-value";
                performance.textContent =
                    `Performance: ${trip.performance || "Not rated"}`;

                
                const events = document.createElement("div");
                events.className = "trip-history-stat-value";
                events.textContent =
                    `Yawns: ${trip.total_yawns || 0} | Eye closures: ${trip.eye_closure_events || 0} | Distractions: ${trip.distraction_events || 0}`;

                
                    stats.append(score, rating, performance, events);
                card.append(id, date, status, stats);
                subviewContent.appendChild(card);
            });

        } catch (error) {
            console.error("Trip History error:", error);
            subviewContent.textContent =
                `Could not load trip history: ${error.message}`;
        }
    });
}

    if (reportsBtn) {
        reportsBtn.addEventListener("click", () => {
            let reports = [];

            try {
                const saved = localStorage.getItem("saferide_last_trip_report");
                if (saved) reports.push(JSON.parse(saved));
            } catch (error) {
                console.warn("Could not read saved trip report:", error);
            }

            if (window.safeRideLastTripReport) {
                reports.unshift(window.safeRideLastTripReport);
            }

            const uniqueReports = reports.filter(
                (report, index, array) =>
                    array.findIndex(item => item.id === report.id) === index
            );

            if (!uniqueReports.length) {
                showSubview(
                    "Trip Reports",
                    '<p class="text-sm text-gray-400">No saved trip report is available yet. Complete a trip first.</p>'
                );
                return;
            }

            
            const reportHtml = uniqueReports.map((report) => {
                const completedAt = report.completedAt
                    ? new Date(report.completedAt).toLocaleString()
                    : "Date unavailable";

                return '<div class="trip-history-item">' +
                    '<div class="trip-history-id">' +
                        (report.id || "Completed trip") +
                    '</div>' +
                    '<div class="trip-history-date">' +
                        completedAt +
                    '</div>' +
                    '<div class="trip-history-stats">' +
                        '<div class="trip-history-stat">' +
                            '<div class="trip-history-stat-label">SCORE</div>' +
                            '<div class="trip-history-stat-value">' +
                                (report.score ?? "--") + '/10' +
                            '</div>' +
                        '</div>' +
                        '<div class="trip-history-stat">' +
                            '<div class="trip-history-stat-label">RATING</div>' +
                            '<div class="trip-history-stat-value">' +
                                (report.stars || "--") +
                            '</div>' +
                        '</div>' +
                        '<div class="trip-history-stat">' +
                            '<div class="trip-history-stat-label">DURATION</div>' +
                            '<div class="trip-history-stat-value">' +
                                (report.duration || "--") +
                            '</div>' +
                        '</div>' +
                    '</div>' +
                '</div>';
            }).join("");

            showSubview("Trip Reports", reportHtml);
        });
    }
})();






// SafeRide AI — permanent Profile button connection
(() => {
    const profileButton = document.getElementById("profile-btn");
    const profileModal = document.getElementById("profile-modal");
    const closeButton = document.getElementById("profile-close-btn");
    const backdrop = document.getElementById("profile-backdrop");

    if (!profileButton || !profileModal) {
        console.error("SafeRide Profile: button or modal not found.");
        return;
    }

    profileButton.onclick = () => {
        profileModal.classList.remove("hidden");
        profileModal.style.display = "flex";
        document.body.style.overflow = "hidden";

        if (window.lucide) {
            window.lucide.createIcons();
        }
    };

    const closeProfile = () => {
        profileModal.classList.add("hidden");
        profileModal.style.display = "";
        document.body.style.overflow = "";
    };

    if (closeButton) {
        closeButton.onclick = closeProfile;
    }

    if (backdrop) {
        backdrop.onclick = closeProfile;
    }

    console.log("SafeRide Profile: click handlers connected.");
})();

 // SafeRide AI — Safe Stop Maps
document.addEventListener("DOMContentLoaded", () => {
    const mapsTile = document.getElementById("safe-stop-map");

    if (!mapsTile) {
        console.warn("Safe Stop Maps tile not found.");
        return;
    }

    
mapsTile.addEventListener("click", async () => {
    const choice = prompt(
        "Find a safe stop:\n1. Petrol Pumps\n2. Hospitals\n3. Police Stations\n4. Rest Areas",
        "1"
    );

    const categories = {
        "1": "petrol",
        "2": "hospital",
        "3": "police",
        "4": "rest"
    };

    const category = categories[choice];
    if (!category) return;

    // Default coordinates: Noida
    let latitude = 28.6692;
    let longitude = 77.4538;

    // Use the driver's current location when permission is granted.
    if (navigator.geolocation) {
        try {
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, {
                    enableHighAccuracy: true,
                    timeout: 5000
                });
            });

            latitude = position.coords.latitude;
            longitude = position.coords.longitude;
        } catch (error) {
            console.warn("Location unavailable; using default coordinates.");
        }
    }

    try {
        const response = await fetch(
            `http://127.0.0.1:8000/safe-stops?category=${encodeURIComponent(category)}&latitude=${latitude}&longitude=${longitude}`
        );

        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }

        const data = await response.json();

        if (data.success && data.maps_url) {
            window.open(data.maps_url, "_blank", "noopener,noreferrer");
        } else {
            alert(data.message || "Could not find safe stops.");
        }
    } catch (error) {
        console.error("Safe Stop API error:", error);
        alert("Safe Stop API se connect nahi ho paaya. Check that the backend is running.");
    }
});

});

// ============================================================
// SAFETY TRENDS + DRIVER INSIGHTS
// ============================================================

async function loadDriverInsights() {
    const panel = document.getElementById("driver-insights-panel");
    const list = document.getElementById("driver-insights-list");
    const driverId = window.safeRideUser?.driver_id || DRIVER_ID;

    if (!driverId || !panel) {
        console.warn("Driver ID or insights panel is unavailable.");
        return;
    }

    panel.classList.remove("hidden");

    if (list) {
        list.innerHTML = "<li>Loading your safety insights...</li>";
    }

    try {
        const response = await fetch(
            `${BACKEND_URL}/drivers/${encodeURIComponent(driverId)}/history`
        );

        if (!response.ok) {
            throw new Error("Could not load trip history.");
        }

        const data = await response.json();
        const trips = (data.trips || [])
            .filter(trip =>
                trip.final_score !== null &&
                trip.final_score !== undefined &&
                Number.isFinite(Number(trip.final_score))
            )
            .reverse();

        document.getElementById("insights-total-trips").textContent =
            data.total_trips ?? trips.length;

        const scores = trips.map(trip =>
            Math.max(0, Math.min(10, Number(trip.final_score)))
        );
        
        const totalYawns = trips.reduce(
            (sum, trip) => sum + Number(trip.total_yawns || 0), 0
        );

        const totalEyeEvents = trips.reduce(
            (sum, trip) => sum + Number(trip.eye_closure_events || 0), 0
        );

        const totalDistractions = trips.reduce(
            (sum, trip) => sum + Number(trip.distraction_events || 0), 0
        );

        document.getElementById("insights-total-yawns").textContent = totalYawns;
        document.getElementById("insights-eye-events").textContent = totalEyeEvents;
        document.getElementById("insights-distraction-events").textContent = totalDistractions;


        const average = scores.length
            ? scores.reduce((sum, score) => sum + score, 0) / scores.length
            : null;
        
        const averageLabel = document.getElementById("insights-average-score");

        if (averageLabel) {
            averageLabel.title = "Average of recorded trip safety scores (out of 10)";
            averageLabel.textContent =
                average === null ? "-- / 10" : `${average.toFixed(1)} / 10`;
            }
        document.getElementById("insights-average-score").textContent =
            average === null ? "-- / 10" : `${average.toFixed(1)} / 10`;

        document.getElementById("insights-latest-score").textContent =
            scores.length ? `${scores[scores.length - 1].toFixed(1)} / 10` : "-- / 10";

        const insights = [];
        
        if (scores.length > 0) {
            const latestScore = scores[scores.length - 1];

            if (latestScore >= 9) {
                insights.push("Great performance! Maintain your safe-driving habits.");
            } else if (latestScore >= 7) {
                insights.push("Good performance. Keep working toward a consistently high safety score.");
            } else if (latestScore >= 5) {
                insights.push("Focus on reducing fatigue and distraction alerts to improve your score.");
            } else {
                insights.push("Your latest safety score is low. Prioritize rest and avoid driving while tired or distracted.");
            }
        }


        if (!trips.length) {
            insights.push("Complete and end a trip to see your safety score trend.");
        } else {
            const recent = trips.slice(-3);
            const previous = trips.slice(-6, -3);

            const recentAvg = recent.reduce(
                (sum, trip) => sum + Number(trip.final_score), 0
            ) / recent.length;

            if (previous.length) {
                const previousAvg = previous.reduce(
                    (sum, trip) => sum + Number(trip.final_score), 0
                ) / previous.length;

                if (recentAvg > previousAvg + 0.2) {
                    insights.push("Your recent safety scores are improving. Keep up the safe-driving habits.");
                } else if (recentAvg < previousAvg - 0.2) {
                    insights.push("Your recent scores have declined. Review your recent alerts and focus on safer driving.");
                } else {
                    insights.push("Your recent safety scores are broadly stable.");
                }
            } else {
                insights.push("Complete more trips to reveal longer-term safety trends.");
            }

            const yawns = trips.reduce(
                (sum, trip) => sum + Number(trip.total_yawns || 0), 0
            );
            const eyeEvents = trips.reduce(
                (sum, trip) => sum + Number(trip.eye_closure_events || 0), 0
            );
            const distractions = trips.reduce(
                (sum, trip) => sum + Number(trip.distraction_events || 0), 0
            );

            if (yawns > 0) {
                insights.push(`Yawning was recorded ${yawns} time(s). Take breaks when tired.`);
            }

            if (eyeEvents > 0) {
                insights.push(`Eye-closure events recorded: ${eyeEvents}. Never continue driving when drowsy.`);
            }

            if (distractions > 0) {
                insights.push(`Distraction events recorded: ${distractions}. Keep your attention on the road.`);
            }

            if (yawns === 0 && eyeEvents === 0 && distractions === 0) {
                insights.push("No yawns, eye-closure events or distractions are recorded in the available trip history.");
            }
        }

        if (list) {
            list.replaceChildren();

            insights.forEach(message => {
                const item = document.createElement("li");
                item.textContent = message;
                list.appendChild(item);
            });
        }
        
        const concernValue = document.getElementById("insights-top-concern");
        const concernDetail = document.getElementById("insights-top-concern-detail");

        if (concernValue && concernDetail) {
            const concerns = [
                { name: "Yawning", count: totalYawns, detail: "Consider taking breaks when tired." },
                { name: "Eye Closure", count: totalEyeEvents, detail: "Avoid driving when drowsy." },
                { name: "Distraction", count: totalDistractions, detail: "Keep your attention on the road." }
            ];

            concerns.sort((a, b) => b.count - a.count);

            if (concerns[0].count === 0) {
                concernValue.textContent = "No events recorded";
                concernDetail.textContent = "No yawns, eye-closure events or distractions are recorded in the available trip history.";
            } else {
                concernValue.textContent = concerns[0].name;
                concernDetail.textContent =
                    `${concerns[0].count} event(s) recorded. ${concerns[0].detail}`;
            }
        }


        drawSafetyTrend(scores);

    } catch (error) {
        console.error("Driver insights error:", error);

        if (list) {
            list.innerHTML =
                "<li>Unable to load insights. Check that the backend is running and try again.</li>";
        }
    }
}

function drawSafetyTrend(scores) {
    const canvas = document.getElementById("safety-trend-chart");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.clientWidth || 500;
    const height = 180;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = "100%";
    canvas.style.height = `${height}px`;

    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const left = 32;
    const right = 12;
    const top = 16;
    const bottom = 28;
    const chartWidth = width - left - right;
    const chartHeight = height - top - bottom;

    ctx.font = "11px sans-serif";
    ctx.fillStyle = "#9ca3af";
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 1;

    [0, 5, 10].forEach(value => {
        const y = top + chartHeight * (1 - value / 10);
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(width - right, y);
        ctx.stroke();
        ctx.fillText(String(value), 8, y + 4);
    });

    if (!scores.length) {
        ctx.fillStyle = "#9ca3af";
        ctx.fillText("No completed trip scores available yet.", left + 8, top + chartHeight / 2);
        return;
    }

    const points = scores.map((score, index) => ({
        x: scores.length === 1
            ? left + chartWidth / 2
            : left + index * chartWidth / (scores.length - 1),
        y: top + chartHeight * (1 - score / 10)
    }));

    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 2;
    ctx.beginPath();

    points.forEach((point, index) => {
        if (index === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
    });

    ctx.stroke();

    points.forEach((point, index) => {
        ctx.beginPath();
        ctx.fillStyle = "#22c55e";
        ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#d1d5db";
        ctx.fillText(String(index + 1), point.x - 3, height - 8);
    });
}