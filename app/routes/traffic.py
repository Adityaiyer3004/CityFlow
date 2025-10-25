import os
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/summary")
def get_summary():
    """Return latest AI summary from file"""
    try:
        with open("latest_summary.txt", "r") as f:
            summary = f.read().strip()
        return {"summary": summary}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Summary file not found")
