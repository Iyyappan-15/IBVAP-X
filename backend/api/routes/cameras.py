from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.db.session import get_db
from backend.db.repository import Repository

router = APIRouter(prefix="/cameras", tags=["Cameras"])

@router.get("", status_code=status.HTTP_200_OK)
def get_cameras(db: Session = Depends(get_db)):
    repo = Repository(db)
    cameras = repo.list_cameras()
    return {
        "cameras": [
            {
                "camera_id": c.camera_id,
                "name": c.name,
                "location": c.location_description,
                "latitude": c.latitude,
                "longitude": c.longitude,
                "status": c.status
            }
            for c in cameras
        ]
    }

@router.get("/{camera_id}", status_code=status.HTTP_200_OK)
def get_camera_detail(camera_id: str, db: Session = Depends(get_db)):
    repo = Repository(db)
    cam = repo.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")
    return {
        "camera_id": cam.camera_id,
        "name": cam.name,
        "location": cam.location_description,
        "status": cam.status
    }
