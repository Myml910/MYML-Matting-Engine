from __future__ import annotations

from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.core.image_io import encode_png, load_image_from_bytes
from app.core.model_router import get_backend
from app.schemas.request import MattingMode, ModelName


router = APIRouter(tags=["matting"])


@router.post("/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    model: ModelName = Form(default="dummy"),
    mode: MattingMode = Form(default="rgba"),
    output_mask: bool = Form(default=False),
    edge_radius: int = Form(default=8),
    blend: float = Form(default=1.0),
) -> StreamingResponse:
    settings = get_settings()
    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit")

    try:
        image = load_image_from_bytes(data)
        backend = get_backend(model)
        result = backend.predict(image, edge_radius=edge_radius, blend=blend)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to process image") from exc

    output_image = result.mask if output_mask or mode == "mask" else result.rgba
    png_bytes = encode_png(output_image)
    filename_stem = file.filename.rsplit(".", 1)[0] if file.filename else "matting"
    suffix = "mask" if output_mask or mode == "mask" else "rgba"
    headers = {"Content-Disposition": f'inline; filename="{filename_stem}_{model}_{suffix}.png"'}
    return StreamingResponse(BytesIO(png_bytes), media_type="image/png", headers=headers)
