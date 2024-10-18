import sys

from fastapi import APIRouter, Request

sys.path.append("..")
from schemas import FreeFormJSON

router = APIRouter(prefix="/whisper", tags=["whisper.cpp"])

@router.get("/inference")
def audio_to_text(request: Request):
    pass # for display endpoint in swagger

@router.get("/load")
def load_model(request: Request):
    pass # for display endpoint in swagger

