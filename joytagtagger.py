# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags

import comfy.utils
import asyncio
import aiohttp
import numpy as np
import os
import sys
import torch.amp.autocast_mode
from .Models import VisionModel
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from server import PromptServer
from aiohttp import web
from .utils import get_ext_dir, get_comfy_dir, download_to_file, update_node_status, wait_for_async, get_extension_config, log
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
IMAGE_SIZE = 448

config = get_extension_config()

defaults = {
    "model": "joytag",
    "threshold": 0.35,
    "replace_underscore": False,
    "trailing_comma": False,
    "exclude_tags": "",
    "HF_ENDPOINT": "https://huggingface.co",
    "model_files": "model.safetensors, top_tags.txt, config.json"
}
defaults.update(config.get("settings", {}))

models_dir = get_ext_dir("models", mkdir=True)
model_files = config["model_files"].split(', ')


def prepare_image(image):
    # Pad image to square
    w, h = image.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2

    padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # Resize image
    if max_dim != IMAGE_SIZE:
        padded_image = padded_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    return padded_image


async def tag(image, threshold=0.35, exclude_tags="", replace_underscore=True, trailing_comma=False, client_id=None, node=None):
    if not any(model_files):
        await download_model(client_id, node)

    model = VisionModel.load_model(models_dir)
    model.eval()
    model = model.to('cuda')

    with open(os.path.join(models_dir, 'top_tags.txt'), "r", encoding="utf-8") as f:
        top_tags = [line.strip() for line in f.readlines() if line.strip()]

    image = prepare_image(image)
    tensor = TVF.pil_to_tensor(image) / 255.0
    # Normalize
    tensor = TVF.normalize(tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    batch = {
        'image': tensor.unsqueeze(0).to('cuda'),
    }

    with torch.amp.autocast_mode.autocast('cuda', enabled=True):
        preds = model(batch)
        tag_preds = preds['tags'].sigmoid().cpu()

    scores = {top_tags[i]: tag_preds[0][i] for i in range(len(top_tags))}
    all = []
    for tag, score in scores.items():
        if score >= threshold:
            if replace_underscore and len(tag) > 3:
                tag = tag.replace("_", " ")
            if tag not in exclude_tags:
                all.append(tag)

    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    all = [tag for tag in all if tag not in remove]

    res = ("" if trailing_comma else ", ").join((item.replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for item in all))

    print(res)
    return res


async def download_model(client_id, node):
    hf_endpoint = os.getenv("HF_ENDPOINT", defaults["HF_ENDPOINT"])
    if not hf_endpoint.startswith("https://"):
        hf_endpoint = f"https://{hf_endpoint}"
    if hf_endpoint.endswith("/"):
        hf_endpoint = hf_endpoint.rstrip("/")

    url = config["repo"]
    url = url.replace("{HF_ENDPOINT}", hf_endpoint)
    url = f"{url}/resolve/main/"
    async with aiohttp.ClientSession(loop=asyncio.get_event_loop()) as session:
        async def update_callback(perc):
            nonlocal client_id
            message = ""
            if perc < 100:
                message = "Downloading Joytag model"
            update_node_status(client_id, node, message, perc)

        try:
            await download_to_file(
                f"{url}model.safetensors", os.path.join(models_dir, f"model.safetensors"), update_callback, session=session)
            await download_to_file(
                f"{url}top_tags.txt", os.path.join(models_dir, "top_tags.txt"), update_callback, session=session)
            await download_to_file(
                f"{url}config.json", os.path.join(models_dir, "config.json"), update_callback, session=session)
        except aiohttp.client_exceptions.ClientConnectorError as err:
            log("Unable to download model. Download files manually or try using a HF mirror/proxy website by setting the environment variable HF_ENDPOINT=https://.....", "ERROR", True)
            raise

        update_node_status(client_id, node, None)

    return web.Response(status=200)


@PromptServer.instance.routes.get("/zakack/joytagtagger/tag")
async def get_tags(request):
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)

    type = request.query.get("type", "output")
    if type not in ["output", "input", "temp"]:
        return web.Response(status=400)

    target_dir = get_comfy_dir(type)
    image_path = os.path.abspath(os.path.join(
        target_dir, request.query.get("subfolder", ""), request.query["filename"]))
    c = os.path.commonpath((image_path, target_dir))
    if os.path.commonpath((image_path, target_dir)) != target_dir:
        return web.Response(status=403)

    if not os.path.isfile(image_path):
        return web.Response(status=404)

    image = Image.open(image_path)

    return web.json_response(await tag(image, client_id=request.rel_url.query.get("clientId", ""), node=request.rel_url.query.get("node", "")))


class JoytagTagger:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", ),
            "threshold": ("FLOAT", {"default": defaults["threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "replace_underscore": ("BOOLEAN", {"default": defaults["replace_underscore"]}),
            "trailing_comma": ("BOOLEAN", {"default": defaults["trailing_comma"]}),
            "exclude_tags": ("STRING", {"default": defaults["exclude_tags"]}),
        }}

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def tag(self, image, threshold, exclude_tags="", replace_underscore=False, trailing_comma=False):
        tensor = image*255
        tensor = np.array(tensor, dtype=np.uint8)

        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        tags = []
        for i in range(tensor.shape[0]):
            image = Image.fromarray(tensor[i])
            tags.append(wait_for_async(lambda: tag(image, threshold, exclude_tags, replace_underscore, trailing_comma)))
            pbar.update(1)
        return {"ui": {"tags": tags}, "result": (tags,)}


NODE_CLASS_MAPPINGS = {
    "JoytagTagger|zakack": JoytagTagger,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "JoytagTagger|zakack": "Joytag Tagger",
}
