from typing import List, Literal, Any

from pydantic import BaseModel
from openai.types import Model

class CustomModel(Model):
    type: Literal["text-generation", "text-embeddings-inference", "image-text-inference", "audio-text-inference"]


class Models(BaseModel):
    object: str
    data: List[CustomModel]


class FreeFormJSON(BaseModel):
    Any
