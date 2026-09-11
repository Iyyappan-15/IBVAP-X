from fastapi import APIRouter, status

router = APIRouter(prefix="/analysis", tags=["Analysis"])

@router.post("/start", status_code=status.HTTP_200_OK)
def start_analysis():
    return {"status": "started", "message": "IBVAP-X analysis pipeline active."}

@router.post("/stop", status_code=status.HTTP_200_OK)
def stop_analysis():
    return {"status": "stopped", "message": "IBVAP-X analysis pipeline paused."}
