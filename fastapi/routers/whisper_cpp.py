import sys

from fastapi import APIRouter, Request

sys.path.append("..")
from schemas import FreeFormJSON

router = APIRouter(prefix="/whisper", tags=["whisper.cpp"])

@router.get("/inference")
def health_check(request: Request):
    pass # for display endpoint in swagger

@router.get("/load")
def get_models(request: Request):
    pass # for display endpoint in swagger

