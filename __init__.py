import io
import json
import requests
import torch
import numpy as np
from PIL import Image
import comfy.utils

_models_cache = {"models": [], "server": ""}

def get_remote_models(server_url):
    global _models_cache
    if _models_cache["server"] != server_url or not _models_cache["models"]:
        try:
            response = requests.get(f"{server_url}/models", timeout=5)
            if response.status_code == 200:
                _models_cache["models"] = response.json().get("models", [])
                _models_cache["server"] = server_url
        except:
            pass
    return _models_cache["models"] if _models_cache["models"] else ["(server unavailable)"]


class RemoteUpscaleImage:
    @classmethod
    def INPUT_TYPES(cls):
        default_server = "http://localhost:8189"
        models = get_remote_models(default_server)

        return {
            "required": {
                "image": ("IMAGE",),
                "server_url": ("STRING", {"default": default_server}),
                "model_name": (models, {"default": models[0] if models else ""}),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "show_progress": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        server_url = kwargs.get("server_url", "http://localhost:8189")
        get_remote_models(server_url)
        return float("nan")

    def upscale(self, image, server_url, model_name, tile_size, show_progress):
        if model_name == "(server unavailable)":
            raise RuntimeError("Remote upscale server is not available")

        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        if show_progress:
            return self._upscale_with_progress(server_url, model_name, tile_size, img_bytes)
        else:
            return self._upscale_simple(server_url, model_name, tile_size, img_bytes)

    def _upscale_simple(self, server_url, model_name, tile_size, img_bytes):
        try:
            response = requests.post(
                f"{server_url}/upscale",
                files={"file": ("image.png", img_bytes, "image/png")},
                data={"model_name": model_name, "tile_size": tile_size},
                timeout=600
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Remote upscale failed: {e}")

        result_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)
        return (result_tensor,)

    def _upscale_with_progress(self, server_url, model_name, tile_size, img_bytes):
        try:
            response = requests.post(
                f"{server_url}/upscale_stream",
                files={"file": ("image.png", img_bytes, "image/png")},
                data={"model_name": model_name, "tile_size": tile_size},
                timeout=600,
                stream=True
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Remote upscale failed: {e}")

        pbar = None
        result_id = None
        last_progress = 0

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if pbar is None and 'total' in data:
                        pbar = comfy.utils.ProgressBar(data['total'])

                    if pbar and 'progress' in data:
                        progress = data['progress']
                        if progress > last_progress:
                            pbar.update(progress - last_progress)
                            last_progress = progress

                    if data.get('status') == 'done' and 'result_id' in data:
                        result_id = data['result_id']

        if result_id is None:
            raise RuntimeError("No result ID received from server")

        # Fetch the result image
        try:
            result_response = requests.get(f"{server_url}/result/{result_id}", timeout=60)
            result_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch result: {e}")

        result_image = Image.open(io.BytesIO(result_response.content)).convert("RGB")
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)
        return (result_tensor,)


class RefreshRemoteModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"default": "http://localhost:8189"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "refresh"
    CATEGORY = "image/upscaling"

    def refresh(self, server_url):
        global _models_cache
        _models_cache = {"models": [], "server": ""}
        models = get_remote_models(server_url)
        return (f"Found {len(models)} models: {', '.join(models)}",)


NODE_CLASS_MAPPINGS = {
    "RemoteUpscaleImage": RemoteUpscaleImage,
    "RefreshRemoteModels": RefreshRemoteModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteUpscaleImage": "Remote Upscale",
    "RefreshRemoteModels": "Refresh Remote Models",
}
