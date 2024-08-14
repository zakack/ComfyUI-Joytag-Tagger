# ComfyUI Joytag Tagger

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension allowing the interrogation of booru tags from images.

Based on [SmilingWolf/wd-v1-4-tags](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags) and [toriato/stable-diffusion-webui-Joytag-tagger](https://github.com/toriato/stable-diffusion-webui-WD14-tagger)  
All models created by [fancyfeast](https://huggingface.co/fancyfeast/joytag)

## Installation
1. `git clone https://github.com/zakack/ComfyUI-Joytag-Tagger` into the `custom_nodes` folder 
    - e.g. `custom_nodes\ComfyUI-Joytag-Tagger`  
2. Open a Command Prompt/Terminal/etc
3. Change to the `custom_nodes\ComfyUI-Joytag-Tagger` folder you just created 
    - e.g. `cd C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Joytag-Tagger` or wherever you have it installed
4.  Install python packages
      - **Windows Standalone installation** (embedded python):   
       `../../../python_embeded/python.exe -s -m pip install -r requirements.txt`  
      - **Manual/non-Windows installation**   
        `pip install -r requirements.txt`

## Usage
Add the node via `image` -> `JoytagTagger|zakack`  
![image](https://github.com/zakack/ComfyUI-Joytag-Tagger/assets/125205205/ee6756ae-73f6-4e9f-a3da-eb87a056eb87)  
Models are automatically downloaded at runtime if missing.  
![image](https://github.com/zakack/ComfyUI-Joytag-Tagger/assets/125205205/cc09ae71-1a38-44da-afec-90f470a4b47d)  
Supports tagging and outputting multiple batched inputs.   
- **threshold**: The score for the tag to be considered valid
- **exclude_tags** A comma separated list of tags that should not be included in the results

Quick interrogation of images is also available on any node that is displaying an image, e.g. a `LoadImage`, `SaveImage`, `PreviewImage` node.  
Simply right click on the node (or if displaying multiple images, on the image you want to interrogate) and select `Joytag Tagger` from the menu  
![image](https://github.com/zakack/ComfyUI-Joytag-Tagger/assets/125205205/11733899-6163-49f6-a22b-8dd86d910de6)

Settings used for this are in the `settings` section of `zakack.json`.

### Offline Use
Simplest way is to use it online, interrogate an image, and the model will be downloaded and cached, however if you want to manually download the models:
- Create a `models` folder (in same folder as the `joytagtagger.py`)
- From the `fancyfeast/joytag` repo on Huggingface:
- Download `model.safetensors`
- Download `top_tags.txt`
- Download `config.json`

## Changelog
- 2024-08-14 - Moved to own repo, changed from ONNX to PyTorch, removed multiple models support
