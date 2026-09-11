from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.db.repository import Repository
from backend.interfaces import AlertState
from backend.alerts.acknowledgement import OperatorAcknowledgementManager

router = APIRouter(prefix="/alerts", tags=["Alerts"])

class ActionRequest(BaseModel):
    operator_username: str = "bop_operator"
    notes: Optional[str] = None

@router.get("", status_code=status.HTTP_200_OK)
def get_alerts(db: Session = Depends(get_db)):
    repo = Repository(db)
    alerts = repo.list_alerts(limit=50)
    return {
        "alerts": [
            {
                "alert_id": a.alert_id,
                "camera_id": a.camera_id,
                "event_priority": a.event_priority,
                "event_priority_score": a.event_priority_score,
                "camera_reliability": a.camera_reliability,
                "actionability": a.actionability,
                "state": a.state
            }
            for a in alerts
        ]
    }

@router.get("/{alert_id}", status_code=status.HTTP_200_OK)
def get_alert_detail(alert_id: str, db: Session = Depends(get_db)):
    repo = Repository(db)
    alert = repo.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
    return alert

@router.post("/{alert_id}/acknowledge", status_code=status.HTTP_200_OK)
def acknowledge_alert(alert_id: str, req: ActionRequest, db: Session = Depends(get_db)):
    repo = Repository(db)
    alert = repo.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")

    ack_mgr = OperatorAcknowledgementManager()
    action = ack_mgr.record_action(alert_id, req.operator_username, AlertState.ACKNOWLEDGED, req.notes)
    repo.update_alert_state(alert_id, AlertState.ACKNOWLEDGED)

    return {"status": "ok", "action_id": action.action_id, "new_state": "ACKNOWLEDGED"}
