from fastapi import APIRouter, status
from backend.coverage.coverage_engine import GeometricCoverageEngine

router = APIRouter(prefix="/coverage", tags=["Coverage"])

@router.get("", status_code=status.HTTP_200_OK)
def get_coverage():
    engine = GeometricCoverageEngine()
    result = engine.compute_coverage()
    return result
