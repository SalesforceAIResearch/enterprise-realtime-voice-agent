"""
Chapter 8: Enterprise Tool Definitions

Defines the tools (functions) that the voice agent can call.
Each tool has:
  - A schema (name, description, parameters) for the LLM
  - A handler function that executes the action
  - Mock data for demo purposes

In production, handlers would call real APIs/databases.
"""

import json
from datetime import datetime

# ---------------------------------------------------------------------------
# Mock data (simulating a hospital database)
# ---------------------------------------------------------------------------
DOCTORS = {
    "Dr. Johnson": {"specialty": "General Medicine", "department": "Internal Medicine", "phone": "(555) 100-1001"},
    "Dr. Patel": {"specialty": "Cardiology", "department": "Cardiology", "phone": "(555) 100-1002"},
    "Dr. Lee": {"specialty": "Orthopedics", "department": "Orthopedics", "phone": "(555) 100-1003"},
    "Dr. Garcia": {"specialty": "Pediatrics", "department": "Pediatrics", "phone": "(555) 100-1004"},
}

PATIENTS = {
    "John Smith": {"id": "P-1001", "phone": "(555) 200-2001", "insurance": "BlueCross Premier", "dob": "1985-03-15"},
    "Jane Doe": {"id": "P-1002", "phone": "(555) 200-2002", "insurance": "Aetna Gold", "dob": "1990-07-22"},
    "Emily Davis": {"id": "P-1003", "phone": "(555) 200-2003", "insurance": "United Health", "dob": "1978-11-08"},
}

APPOINTMENTS = []


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function calling format)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check a doctor's available appointment slots for a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string", "description": "Doctor's name (e.g., 'Dr. Johnson')"},
                    "date": {"type": "string", "description": "Date to check (YYYY-MM-DD format)"},
                },
                "required": ["doctor_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_appointment",
            "description": "Book an appointment for a patient with a doctor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string", "description": "Patient's full name"},
                    "doctor_name": {"type": "string", "description": "Doctor's name"},
                    "date": {"type": "string", "description": "Appointment date (YYYY-MM-DD)"},
                    "time": {"type": "string", "description": "Appointment time (e.g., '3:00 PM')"},
                    "reason": {"type": "string", "description": "Reason for visit"},
                },
                "required": ["patient_name", "doctor_name", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel an existing appointment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "string", "description": "The appointment ID to cancel"},
                },
                "required": ["appointment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_info",
            "description": "Look up patient information by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string", "description": "Patient's full name"},
                },
                "required": ["patient_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_doctor_info",
            "description": "Look up doctor information including specialty and availability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string", "description": "Doctor's name"},
                },
                "required": ["doctor_name"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------
def check_availability(doctor_name: str, date: str = None) -> dict:
    if doctor_name not in DOCTORS:
        return {"error": f"Doctor '{doctor_name}' not found. Available: {', '.join(DOCTORS.keys())}"}
    date = date or datetime.now().strftime("%Y-%m-%d")
    return {
        "doctor": doctor_name,
        "date": date,
        "available_slots": [
            {"time": "9:00 AM", "duration": "30 min"},
            {"time": "10:30 AM", "duration": "30 min"},
            {"time": "2:00 PM", "duration": "30 min"},
            {"time": "3:30 PM", "duration": "30 min"},
        ],
    }


def schedule_appointment(patient_name: str, doctor_name: str, date: str, time: str, reason: str = "") -> dict:
    if doctor_name not in DOCTORS:
        return {"error": f"Doctor '{doctor_name}' not found"}
    apt_id = f"APT-{len(APPOINTMENTS) + 1001}"
    apt = {
        "id": apt_id, "patient": patient_name, "doctor": doctor_name,
        "date": date, "time": time, "reason": reason, "status": "confirmed",
        "location": "Main Hospital, Room 204",
    }
    APPOINTMENTS.append(apt)
    return apt


def cancel_appointment(appointment_id: str) -> dict:
    for apt in APPOINTMENTS:
        if apt["id"] == appointment_id:
            apt["status"] = "cancelled"
            return {"status": "cancelled", "appointment_id": appointment_id}
    return {"error": f"Appointment '{appointment_id}' not found"}


def get_patient_info(patient_name: str) -> dict:
    info = PATIENTS.get(patient_name)
    if not info:
        return {"error": f"Patient '{patient_name}' not found. Known patients: {', '.join(PATIENTS.keys())}"}
    return {"name": patient_name, **info}


def get_doctor_info(doctor_name: str) -> dict:
    info = DOCTORS.get(doctor_name)
    if not info:
        return {"error": f"Doctor '{doctor_name}' not found. Available: {', '.join(DOCTORS.keys())}"}
    return {"name": doctor_name, **info}


TOOL_HANDLERS = {
    "check_availability": check_availability,
    "schedule_appointment": schedule_appointment,
    "cancel_appointment": cancel_appointment,
    "get_patient_info": get_patient_info,
    "get_doctor_info": get_doctor_info,
}


SYSTEM_PROMPT = """You are a helpful medical receptionist for City General Hospital. You can:
- Check doctor availability and schedule appointments
- Look up patient and doctor information
- Cancel appointments

Be concise and friendly. When a patient asks to schedule, check availability first.
Always confirm details before booking. Use 2-3 sentences maximum per response."""
