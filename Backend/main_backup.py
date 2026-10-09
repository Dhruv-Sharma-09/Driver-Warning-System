from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4
from database import engine, Base
import models


# ============================================================
# SAFE RIDE AI - FASTAPI BACKEND
# ============================================================

app = FastAPI(
    title="SafeRide AI API",
    description="Backend API for the SafeRide AI driver safety platform.",
    version="1.0.0"
)
# Create database tables
Base.metadata.create_all(bind=engine)


# ============================================================
# TEMPORARY IN-MEMORY STORAGE
# ============================================================

drivers = {}
trips = {}
alerts = {}


# ============================================================
# DATA MODELS
# ============================================================

class DriverCreate(BaseModel):
    driver_id: str
    name: str
    phone: str = ""
    vehicle_number: str = ""


class TripStart(BaseModel):
    driver_id: str


class SafetyData(BaseModel):
    eye_closure_duration: float = Field(default=0.0, ge=0)
    yawn_count: int = Field(default=0, ge=0)
    distraction_duration: float = Field(default=0.0, ge=0)
    safety_score: float = Field(default=100.0, ge=0, le=100)
    alertness_score: float = Field(default=100.0, ge=0, le=100)


class AlertCreate(BaseModel):
    trip_id: str
    alert_type: str
    message: str
    severity: str = "warning"


# ============================================================
# HOME / HEALTH
# ============================================================

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "SafeRide AI Backend is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# ============================================================
# DRIVER APIs
# ============================================================

@app.post("/drivers")
def create_driver(driver: DriverCreate):

    if driver.driver_id in drivers:
        raise HTTPException(
            status_code=400,
            detail="Driver already exists."
        )

    driver_data = {
        "driver_id": driver.driver_id,
        "name": driver.name,
        "phone": driver.phone,
        "vehicle_number": driver.vehicle_number,
        "total_trips": 0,
        "average_score": 0,
        "created_at": datetime.now().isoformat()
    }

    drivers[driver.driver_id] = driver_data

    return {
        "message": "Driver created successfully",
        "driver": driver_data
    }


@app.get("/drivers/{driver_id}")
def get_driver(driver_id: str):

    if driver_id not in drivers:
        raise HTTPException(
            status_code=404,
            detail="Driver not found."
        )

    return drivers[driver_id]


@app.get("/drivers/{driver_id}/history")
def get_driver_history(driver_id: str):

    if driver_id not in drivers:
        raise HTTPException(
            status_code=404,
            detail="Driver not found."
        )

    driver_trips = [
        trip for trip in trips.values()
        if trip["driver_id"] == driver_id
    ]

    return {
        "driver_id": driver_id,
        "total_trips": len(driver_trips),
        "trips": driver_trips
    }


# ============================================================
# TRIP APIs
# ============================================================

@app.post("/trips/start")
def start_trip(trip_request: TripStart):

    if trip_request.driver_id not in drivers:
        raise HTTPException(
            status_code=404,
            detail="Driver not found."
        )

    trip_id = "TRIP-" + uuid4().hex[:8].upper()

    trip = {
        "trip_id": trip_id,
        "driver_id": trip_request.driver_id,
        "status": "active",
        "start_time": datetime.now().isoformat(),
        "end_time": None,

        "eye_closure_events": 0,
        "total_yawns": 0,
        "distraction_events": 0,
        "warnings": 0,

        "max_eye_closure": 0,
        "max_distraction_duration": 0,

        "safety_score": 100,
        "rating": 5
    }

    trips[trip_id] = trip

    drivers[trip_request.driver_id]["total_trips"] += 1

    return {
        "message": "Trip started successfully",
        "trip": trip
    }


@app.post("/trips/{trip_id}/safety-data")
def update_safety_data(
    trip_id: str,
    data: SafetyData
):

    if trip_id not in trips:
        raise HTTPException(
            status_code=404,
            detail="Trip not found."
        )

    trip = trips[trip_id]

    if trip["status"] != "active":
        raise HTTPException(
            status_code=400,
            detail="Trip is not active."
        )

    # Update maximum values
    trip["max_eye_closure"] = max(
        trip["max_eye_closure"],
        data.eye_closure_duration
    )

    trip["max_distraction_duration"] = max(
        trip["max_distraction_duration"],
        data.distraction_duration
    )

    # Update event counters
    trip["total_yawns"] = max(
        trip["total_yawns"],
        data.yawn_count
    )

    if data.distraction_duration > 0:
        trip["distraction_events"] += 1

    if data.eye_closure_duration >= 8:
        trip["eye_closure_events"] += 1

    # Update safety score
    trip["safety_score"] = data.safety_score

    return {
        "message": "Safety data recorded",
        "trip_id": trip_id,
        "current_safety_score": trip["safety_score"]
    }


@app.post("/trips/{trip_id}/end")
def end_trip(trip_id: str):

    if trip_id not in trips:
        raise HTTPException(
            status_code=404,
            detail="Trip not found."
        )

    trip = trips[trip_id]

    if trip["status"] != "active":
        raise HTTPException(
            status_code=400,
            detail="Trip has already ended."
        )

    trip["status"] = "completed"
    trip["end_time"] = datetime.now().isoformat()

    # --------------------------------------------------------
    # Calculate final score
    # --------------------------------------------------------

    score = 10.0

    # Yawning penalty
    if trip["total_yawns"] >= 5:
        score -= 2.0
    elif trip["total_yawns"] >= 2:
        score -= 1.0

    # Eye closure penalty
    if trip["eye_closure_events"] > 0:
        score -= min(
            trip["eye_closure_events"] * 0.5,
            2.0
        )

    # Distraction penalty
    if trip["distraction_events"] > 0:
        score -= min(
            trip["distraction_events"] * 0.5,
            2.0
        )

    score = max(0, round(score, 1))

    trip["final_score"] = score

    # Convert 10-point score to 5-star rating
    trip["rating"] = round(score / 2, 1)

    # Performance category
    if score >= 9:
        performance = "Excellent"
    elif score >= 7:
        performance = "Good"
    elif score >= 5:
        performance = "Needs Improvement"
    else:
        performance = "High Risk"

    trip["performance"] = performance

    # Update driver's average score
    driver_id = trip["driver_id"]

    completed_trips = [
        t for t in trips.values()
        if t["driver_id"] == driver_id
        and t["status"] == "completed"
    ]

    if completed_trips:
        average = sum(
            t["final_score"]
            for t in completed_trips
        ) / len(completed_trips)

        drivers[driver_id]["average_score"] = round(
            average,
            1
        )

    return {
        "message": "Trip completed successfully",
        "trip_report": trip
    }


@app.get("/trips/{trip_id}/report")
def get_trip_report(trip_id: str):

    if trip_id not in trips:
        raise HTTPException(
            status_code=404,
            detail="Trip not found."
        )

    trip = trips[trip_id]

    if trip["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Trip has not been completed yet."
        )

    return {
        "trip_id": trip["trip_id"],
        "driver_id": trip["driver_id"],
        "start_time": trip["start_time"],
        "end_time": trip["end_time"],

        "final_score": trip["final_score"],
        "rating": trip["rating"],
        "performance": trip["performance"],

        "total_yawns": trip["total_yawns"],
        "eye_closure_events": trip["eye_closure_events"],
        "distraction_events": trip["distraction_events"],
        "warnings": trip["warnings"],

        "max_eye_closure": trip["max_eye_closure"],
        "max_distraction_duration": trip[
            "max_distraction_duration"
        ]
    }


# ============================================================
# ALERT APIs
# ============================================================

@app.post("/alerts")
def create_alert(alert: AlertCreate):

    alert_id = "ALERT-" + uuid4().hex[:8].upper()

    alert_data = {
        "alert_id": alert_id,
        "trip_id": alert.trip_id,
        "alert_type": alert.alert_type,
        "message": alert.message,
        "severity": alert.severity,
        "timestamp": datetime.now().isoformat()
    }

    alerts[alert_id] = alert_data

    # Increase warning counter for the trip
    if alert.trip_id in trips:
        trips[alert.trip_id]["warnings"] += 1

    return {
        "message": "Alert recorded successfully",
        "alert": alert_data
    }


@app.get("/alerts/{trip_id}")
def get_trip_alerts(trip_id: str):

    trip_alerts = [
        alert for alert in alerts.values()
        if alert["trip_id"] == trip_id
    ]

    return {
        "trip_id": trip_id,
        "total_alerts": len(trip_alerts),
        "alerts": trip_alerts
    }