from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app.model import Classifier
from app.schemas import HealthResponse, PredictResponse, Prediction

MODEL_PATH = Path("model/model.pt")
MODEL_VERSION = "fmnist-mobilenet-v1"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup; tear down on shutdown.
    app.state.classifier = Classifier(MODEL_PATH)
    yield
    app.state.classifier = None


app = FastAPI(
    title="Fashion Classifier",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    clf = getattr(app.state, "classifier", None)
    return HealthResponse(
        status="ok" if clf else "loading",
        model_loaded=clf is not None,
        device=str(clf.device) if clf else "unknown",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(0, ge=0, le=10),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported content type: {file.content_type}")

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File too large (>{MAX_BYTES} bytes)")
    if not raw:
        raise HTTPException(400, "Empty file")

    try:
        image = Image.open(BytesIO(raw))
        image.load()          # force decode now so bad files fail here, not inside predict
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not decode image")

    clf: Classifier = app.state.classifier
    label, conf = clf.predict(image)
    topk = None
    if top_k:
        topk = [Prediction(label=l, confidence=c)
                for l, c in clf.predict_topk(image, k=top_k)]

    return PredictResponse(
        prediction=Prediction(label=label, confidence=conf),
        top_k=topk,
        model_version=MODEL_VERSION,
    )