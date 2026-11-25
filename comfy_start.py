# ComfyUI без интерфейса через comfy-cli: ставит ComfyUI, поднимает сервер в фоне, gen() рисует картинку.
# Состояние ядра живёт между вызовами colab exec, поэтому установка — один раз за сессию (см. README).
import glob
import json
import os
import shlex
import subprocess
import time
import urllib.request

from IPython.display import Image, display

WS = "/content/ComfyUI"
COMFY = f"comfy --skip-prompt --workspace={WS}"
os.environ["HF_API_TOKEN"] = open("/content/.hf_token").read().strip()

HF = "https://huggingface.co"
ZI = f"{HF}/Comfy-Org/z_image_turbo/resolve/main/split_files"
MODELS = {  # имя -> [(url, папка в ComfyUI)]
    "sd15": [(f"{HF}/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors", "models/checkpoints")],
    "sdxl": [(f"{HF}/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", "models/checkpoints")],
    "zimage": [
        (f"{ZI}/diffusion_models/z_image_turbo_bf16.safetensors", "models/diffusion_models"),
        (f"{ZI}/text_encoders/qwen_3_4b.safetensors", "models/text_encoders"),
        (f"{ZI}/vae/ae.safetensors", "models/vae"),
    ],
}
SETS = {  # встроенный граф comfy run: SD1.5 и SDXL отличаются чекпойнтом и размером
    "sd15": "--set checkpoint=v1-5-pruned-emaonly-fp16.safetensors",
    "sdxl": "--set checkpoint=sd_xl_base_1.0.safetensors --set width=1024 --set height=1024",
}


def sh(cmd):
    """Запускает команду и печатает вывод построчно: у colab exec таймаут на тишину."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise RuntimeError(f"команда упала: {cmd}")


def up():
    try:
        return urllib.request.urlopen("http://127.0.0.1:8188/system_stats").status == 200
    except OSError:
        return False


def gen(prompt, model="sd15"):
    """Рисует картинку и показывает её; colab exec --output-image сохранит её локально."""
    for url, folder in MODELS[model]:
        if not os.path.exists(f"{WS}/{folder}/{url.rsplit('/', 1)[1]}"):
            sh(f"{COMFY} model download --url {url} --relative-path {folder}")
    t = time.time()
    if model == "zimage":
        # промпт зашит в шаблон (внутри подграфа), поэтому вписываем его в узел CLIPTextEncode
        if not os.path.exists(f"{WS}/z.json"):
            sh(f"{COMFY} templates fetch image_z_image_turbo -o {WS}/z.json")
        wf = json.load(open(f"{WS}/z.json"))
        for node in wf["definitions"]["subgraphs"][0]["nodes"]:
            if node["type"] == "CLIPTextEncode":
                node["widgets_values"][0] = prompt
        json.dump(wf, open("/content/z_run.json", "w"))
        sh(f"{COMFY} run --workflow /content/z_run.json --wait --timeout 1200")
    else:
        sh(f"{COMFY} run --prompt {shlex.quote(prompt)} {SETS[model]} --wait --timeout 1200")
    img = max(glob.glob(f"{WS}/output/*.png"), key=os.path.getmtime)
    print(f"{img} за {time.time() - t:.0f} с")
    display(Image(filename=img))


if not os.path.exists(f"{WS}/main.py"):
    sh("pip install -q comfy-cli")
    sh("comfy --skip-prompt tracking disable")
    sh(f"{COMFY} install --nvidia")
if not up():
    sh(f"{COMFY} launch --background -- --port 8188")
    while not up():
        print(".", end="", flush=True)
        time.sleep(3)
    print(" ComfyUI запущен")
