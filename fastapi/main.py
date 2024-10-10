import requests
import urllib
import time
import os
from typing import Optional, Union, Annotated

from fastapi import FastAPI, Request, HTTPException, Security, Depends, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from routers.embed import router as EmbRouter
from routers.vllm import router as VllmRouter
from schemas import CustomModel, Models, FreeFormJSON

app = FastAPI(title="All In One LLM", version="1.0.1")

# LLM_URL = "http://localhost:8012"
# VLM_URL = "http://localhost:8022"
# EMB_URL = "http://localhost:8112"

LLM_URL = "http://llm-qwen2_5-72b-int4-2gpu:8012"
VLM_URL = "http://vlm-qwen2-vl-7b:8022"
EMB_URL = "http://embed-gte-qwen2-7b:8112"

# auth
auth_scheme = HTTPBearer(scheme_name="API key")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    def check_api_key():
        pass

else:
    def check_api_key(api_key: Annotated[HTTPAuthorizationCredentials, Depends(auth_scheme)]):
        if api_key.scheme != "Bearer":
            raise HTTPException(status_code=403, detail="Invalid authentication scheme")
        if api_key.credentials != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")

        return api_key.credentials


# image upload dir
UPLOAD_DIRECTORY = "./uploaded_images"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
app.mount("/images", StaticFiles(directory=UPLOAD_DIRECTORY), name="images")


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"info": f"Image '{file.filename}' uploaded successfully.", "url": f"/images/{file.filename}"}

@app.get("/images/{filename}")
async def get_image(filename: str):
    file_path = os.path.join(UPLOAD_DIRECTORY, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")


@app.get("/health")
def health_check(request: Request, api_key: str = Security(check_api_key)) -> Response:
    """
    Health check of vLLM model and embedding embeddings.
    """
    response = requests.get(f"{LLM_URL}/health")
    llm_status = response.status_code

    response = requests.get(f"{VLM_URL}/health")
    vlm_status = response.status_code

    response = requests.get(f"{EMB_URL}/health")
    emb_status = response.status_code

    if llm_status == 200 and vlm_status == 200 and emb_status == 200:
        return Response(status_code=200)
    else:
        return Response(status_code=500)


@app.get("/v1/models/{model}", tags=["OpenAI"])
@app.get("/v1/models", tags=["OpenAI"])
def get_models(
    request: Request, model: Optional[str] = None, api_key: str = Security(check_api_key)
) -> Union[Models, CustomModel]:
    """
    Show available models
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    llm_model = requests.get(f"{LLM_URL}/v1/models", headers=headers).json()
    vlm_model = requests.get(f"{VLM_URL}/v1/models", headers=headers).json()
    emb_model = requests.get(f"{EMB_URL}/v1/models", headers=headers).json()

    llm_model_data = {
        "id": llm_model["data"][0]["id"],
        "object": "model",
        "owned_by": "vllm",
        "created": llm_model["data"][0]["created"],
        "type": "text-generation",
    }
    vlm_model_data = {
        "id": vlm_model["data"][0]["id"],
        "object": "model",
        "owned_by": "vllm",
        "created": vlm_model["data"][0]["created"],
        "type": "text-generation",
    }
    emb_model_data = {
        "id": emb_model["data"][0]["id"],
        "object": "model",
        "owned_by": "sglang",
        "created": round(time.time()),
        "type": "text-embeddings-inference",
    }

    if model is not None:
        # support double encoding for model ID with "/" character
        model = urllib.parse.unquote(urllib.parse.unquote(model))
        if model not in [llm_model_data["id"], vlm_model_data["id"], emb_model_data["id"]]:
            raise HTTPException(status_code=404, detail="Model not found")

        if model == llm_model_data["id"]:
            return llm_model_data
        elif model == vlm_model_data["id"]:
            return vlm_model_data
        else:
            return emb_model_data

    response = {"object": "list", "data": [llm_model_data, vlm_model_data, emb_model_data]}

    return response


@app.post("/v1/embeddings", tags=["OpenAI"])
def embeddings(request: FreeFormJSON):
    pass


@app.post("/v1/completions", tags=["OpenAI"])
def completions(request: FreeFormJSON):
    pass


@app.post("/v1/chat/completions", tags=["OpenAI"])
def chat_completions(request: FreeFormJSON):
    pass

# routers
app.include_router(VllmRouter)
app.include_router(EmbRouter)
