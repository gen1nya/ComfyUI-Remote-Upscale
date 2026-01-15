import io
import os
import json
import torch
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse
from PIL import Image
import numpy as np
from spandrel import ModelLoader

app = FastAPI(title="Remote Upscale Server", version="1.0.0")

MODELS_PATH = os.environ.get("UPSCALE_MODELS_PATH", "./models")

loaded_models = {}
results_cache = {}


def get_model(model_name: str):
    model_path = os.path.join(MODELS_PATH, model_name)
    if model_name not in loaded_models:
        print(f"Loading model: {model_path}")
        model = ModelLoader().load_from_file(model_path).eval().cuda()
        loaded_models[model_name] = model
    return loaded_models[model_name]


@app.get("/models")
async def list_models():
    """List available upscale models."""
    models = []
    if os.path.exists(MODELS_PATH):
        for f in os.listdir(MODELS_PATH):
            if f.endswith(('.pth', '.pt', '.safetensors')):
                models.append(f)
    return {"models": sorted(models), "path": MODELS_PATH}


@app.post("/upscale")
async def upscale(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    tile_size: int = Form(512)
):
    """Upscale an image without progress streaming."""
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    img_np = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()

    model = get_model(model_name)

    with torch.no_grad():
        b, c, h, w = img_tensor.shape
        if h * w < tile_size * tile_size * 4:
            output = model(img_tensor)
        else:
            output = tiled_upscale(img_tensor, model, tile_size)

    output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_np)

    img_bytes = io.BytesIO()
    output_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return Response(content=img_bytes.read(), media_type="image/png")


@app.post("/upscale_stream")
async def upscale_stream(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    tile_size: int = Form(512)
):
    """Upscale an image with SSE progress streaming."""
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    img_np = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()

    model = get_model(model_name)
    result_id = str(uuid.uuid4())

    async def generate():
        nonlocal img_tensor
        with torch.no_grad():
            b, c, h, w = img_tensor.shape
            scale = model.scale
            overlap = 32

            tiles_y = max(1, (h + tile_size - overlap - 1) // (tile_size - overlap))
            tiles_x = max(1, (w + tile_size - overlap - 1) // (tile_size - overlap))
            total_tiles = tiles_y * tiles_x

            if h * w < tile_size * tile_size * 4:
                yield f"data: {json.dumps({'progress': 0, 'total': 1})}\n\n"
                output = model(img_tensor)
                yield f"data: {json.dumps({'progress': 1, 'total': 1})}\n\n"
            else:
                out_h, out_w = int(h * scale), int(w * scale)
                output = torch.zeros((b, c, out_h, out_w), device=img_tensor.device)
                count = torch.zeros((b, c, out_h, out_w), device=img_tensor.device)

                current_tile = 0
                for y in range(0, h, tile_size - overlap):
                    for x in range(0, w, tile_size - overlap):
                        y_end = min(y + tile_size, h)
                        x_end = min(x + tile_size, w)

                        tile = img_tensor[:, :, y:y_end, x:x_end]
                        upscaled_tile = model(tile)

                        out_y, out_x = int(y * scale), int(x * scale)
                        out_y_end, out_x_end = int(y_end * scale), int(x_end * scale)

                        output[:, :, out_y:out_y_end, out_x:out_x_end] += upscaled_tile
                        count[:, :, out_y:out_y_end, out_x:out_x_end] += 1

                        current_tile += 1
                        yield f"data: {json.dumps({'progress': current_tile, 'total': total_tiles})}\n\n"

                output = output / count.clamp(min=1)

        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        output_image = Image.fromarray(output_np)

        img_bytes = io.BytesIO()
        output_image.save(img_bytes, format="PNG")
        results_cache[result_id] = img_bytes.getvalue()

        yield f"data: {json.dumps({'status': 'done', 'result_id': result_id})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/result/{result_id}")
async def get_result(result_id: str):
    """Fetch the result image after streaming completes."""
    if result_id not in results_cache:
        return Response(status_code=404, content="Result not found")

    img_data = results_cache.pop(result_id)
    return Response(content=img_data, media_type="image/png")


def tiled_upscale(img, model, tile_size, overlap=32):
    """Process image in tiles to handle large images."""
    b, c, h, w = img.shape
    scale = model.scale

    out_h, out_w = int(h * scale), int(w * scale)
    output = torch.zeros((b, c, out_h, out_w), device=img.device)
    count = torch.zeros((b, c, out_h, out_w), device=img.device)

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)

            tile = img[:, :, y:y_end, x:x_end]
            upscaled_tile = model(tile)

            out_y, out_x = int(y * scale), int(x * scale)
            out_y_end, out_x_end = int(y_end * scale), int(x_end * scale)

            output[:, :, out_y:out_y_end, out_x:out_x_end] += upscaled_tile
            count[:, :, out_y:out_y_end, out_x:out_x_end] += 1

    return output / count.clamp(min=1)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8189)
