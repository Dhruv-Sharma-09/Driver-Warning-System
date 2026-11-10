# 🚗 Driver Warning System

### Your AI Co-Pilot for Safer Driving

> A smartphone-first driver safety system that uses real-time computer vision to detect drowsiness, yawning, and driver distraction during an active trip.

---

## 📌 Overview

**Driver Warning System** is an AI-powered driver safety and surveillance system designed to provide real-time awareness of potentially unsafe driver behavior.

The system uses a smartphone/browser camera to monitor the driver's face during an active trip. Facial landmarks are processed using **MediaPipe Face Landmarker**, while JavaScript-based computer-vision logic derives safety signals such as:

- 👁️ Prolonged eye closure
- 🥱 Yawning
- 👀 Head orientation / distraction
- 🚨 Escalating safety alerts

The detected signals are evaluated to determine the driver's current safety state and provide appropriate warnings.

SafeRide AI also maintains trip-level safety information through a **FastAPI backend and SQLite database**, enabling trip history and safety reports.

---

## 🎯 Problem Statement

Drivers can become unsafe due to:

### 😴 Driver Fatigue & Drowsiness

Prolonged eye closure and fatigue can reduce driver awareness.

### 📱 Driver Distraction

Looking away from the road for extended periods can reduce attention to the driving environment.

### ⚠️ Lack of Real-Time Safety Monitoring

Traditional trip systems primarily focus on transportation and trip management rather than continuous driver-state monitoring.

Driver Warning System addresses this gap by providing a real-time driver safety layer during an active trip.

---

## 💡 Our Solution

Driver Warning System introduces a **booking/trip-based driver safety mode**.

When a trip starts:

```text
Trip Starts
     ↓
Safety Mode Activated
     ↓
Smartphone Camera Monitoring
     ↓
Facial Landmark Detection
     ↓
Feature Extraction
     ├── Eye Closure
     ├── Yawning
     └── Head Orientation
     ↓
Risk / Safety Decision
     ↓
Real-Time Warning / Alert
     ↓
Trip Safety Data & Report
```

---

## 📸 Screenshots & Demo

The Driver Warning System prototype provides a real-time driver monitoring interface with safety alerts, event logging, and trip-level reporting.

### 🏠 Main Interface

![SafeRide AI Home](docs/screenshots/home.png)

### 🚗 Active Trip Monitoring

![Active Trip Monitoring](docs/screenshots/active_trip.png)

### 😴 Drowsiness Detection

![Drowsiness Alert](docs/screenshots/drowsiness_alert.png)

### 🥱 Fatigue Detection

![Fatigue Detection](docs/screenshots/fatigue%20detection.png)

### 👀 Distraction Detection

![Distraction Alert](docs/screenshots/Distraction_alert.png)

### 📋 Safety Event Log

![Safety Event Log](docs/screenshots/safety_event_log.png)

### 📊 Trip Safety Report

![Trip Report](docs/screenshots/trip_report.png)
---

## ✨ Key Features

- **Real-Time Driver Monitoring:** Uses a browser-accessible camera and MediaPipe Face Landmarker.
- **Drowsiness Detection:** Detects prolonged eye closure and generates safety alerts.
- **Yawning Detection:** Tracks yawning events as potential fatigue indicators.
- **Distraction Monitoring:** Uses head orientation to identify potential driver distraction.
- **Escalating Alerts:** Provides warnings based on the detected risk level.
- **Safety Event Log:** Displays safety-related events during a trip.
- **Trip Safety Reports:** Generates a trip-level performance score and rating.
- **Trip History:** Stores completed trip records through the backend.
- **Safe Stop Maps:** Helps drivers find nearby petrol pumps, hospitals, police stations, and rest areas using Google Maps.
- **Driver Profile:** Displays trip statistics and average safety score.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Computer Vision | MediaPipe Face Landmarker |
| Backend API | Python, FastAPI |
| Database | SQLite |
| API Documentation | Swagger UI |
| Safe Stop Navigation | Google Maps search links |

## 🏗️ System Architecture

```text
Smartphone / Browser Camera
            ↓
MediaPipe Face Landmarker
            ↓
JavaScript Safety Analysis
            ↓
Risk Classification & Alerts
            ↓
Trip Report and Safety Metrics
            ↓
FastAPI Backend
            ↓
SQLite Database
            ↓
Trip History and Driver Profile
```

## ⚙️ Setup and Installation

### Prerequisites

- Python installed on your system
- A modern browser with camera access
- VS Code or another code editor

### 1. Clone or Download the Repository

Download the project and open its root folder in VS Code.

### 2. Create and Activate a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Start the Backend

From the project root:

```powershell
python -m uvicorn main:app --reload --app-dir Backend
```

Backend API: `http://127.0.0.1:8000`

Interactive API documentation: `http://127.0.0.1:8000/docs`

### 5. Start the Frontend

Open a second terminal:

```powershell
cd Frontend
python -m http.server 5501
```

Open `http://127.0.0.1:5501/index_updated.html` in your browser.

Allow camera access when prompted.

## 🔐 Configuration and Privacy

- Keep credentials and secrets in environment variables or a local `.env` file.
- Never commit API secrets, client secrets, or session keys to GitHub.
- Camera access requires browser permission and a supported browser environment.
- Configure authentication credentials separately if using Google sign-in.

## ⚠️ Limitations

- This is a prototype; detection thresholds and safety scores are not clinically validated.
- Lighting, camera angle, face visibility, and device performance can affect detection.
- Head orientation and eye-closure signals are indicators, not definitive proof of distraction or fatigue.
- Safety scores are heuristic estimates and should not be treated as certified driver assessments.
- Safe Stop Maps opens Google Maps search results; it does not independently verify the safety or availability of a location.
- The prototype does not replace driver judgment or safe driving practices.

## 🚀 Future Scope

- Improve detection robustness across different lighting conditions and camera angles.
- Calibrate alert thresholds using representative driving data.
- Add optional GPS-based location handling for nearby safe stops.
- Improve trip analytics and historical safety trends.
- Evaluate performance under controlled conditions before real-world deployment.

## 👥 Team

**Team Name:** Last Contendors

---

*SafeRide AI is a prototype intended to demonstrate AI-assisted driver safety monitoring and should not be considered a certified vehicle safety system.*
