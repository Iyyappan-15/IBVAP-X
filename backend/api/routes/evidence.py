from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.db.session import get_db
from backend.db.repository import Repository
from backend.evidence.hashing import EvidenceHasher

router = APIRouter(prefix="/evidence", tags=["Evidence"])

@router.get("/{evidence_id}", status_code=status.HTTP_200_OK)
def get_evidence(evidence_id: str, db: Session = Depends(get_db)):
    repo = Repository(db)
    ev = repo.get_evidence(evidence_id)
    if not ev:
        raise HTTPException(status_code=404, detail=f"Evidence '{evidence_id}' not found")
    return {
        "evidence_id": ev.evidence_id,
        "alert_id": ev.alert_id,
        "camera_id": ev.camera_id,
        "sha256_hash": ev.sha256_hash,
        "is_sealed": ev.is_sealed
    }
