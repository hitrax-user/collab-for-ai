# Модель на VM Colab: llama.cpp (GGUF) или ExLlamaV3 (EXL3). Это реализация «Ядра VM»: файл читает и шлёт только kernel.py,
# каждый раз вместе с вызовом entry(). Состояние ядра живёт между вызовами colab exec, поэтому модель грузится один раз.
# Служебные кадры для kernel.py — строка \x1e + JSON с полем kind: state, stage, ready, begin, tps, error.
import gc
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

TOKEN = "/content/.hf_token"
if os.path.exists(TOKEN):
    # без токена в окружении huggingface_hub ходит в секреты Colab и ждёт таймаута
    os.environ["HF_TOKEN"] = open(TOKEN).read().strip()
CACHE = "/root/.cache/huggingface/hub"
CTX = 8192  # контекст EXL3 и наименьший контекст llama.cpp
PORT = 8181
URL = f"http://127.0.0.1:{PORT}"
LOG = "/content/llama-server.log"
LLAMA_PIN = "b11798"  # запасная сборка llama.cpp, если свежую не скачать
# Контекст и слои на GPU llama.cpp подбирает сам (--fit on по умолчанию, -c не задаём): берёт контекст, на котором
# модель обучена, и ужимает его, пока модель не влезет в видеопамять, но не ниже -fitc. Не влезла и так — уводит
# слои на CPU. Малой модели (GigaChat 10B-A1.8B) достаётся длинный контекст под промпт с инструментами и
# результаты поиска, крупная (27B) остаётся на 8192 и как можно большим числом слоёв на GPU.
# -fa on и KV-кэш в q8_0 ужимают кэш вдвое почти без потерь качества, а -fitt 256 — запас 256 МиБ вместо 1 ГиБ,
# на T4 это ещё 3–4 слоя крупной модели. Каждый слой на CPU режет скорость в разы.
# -lv 4: строки «offloaded N/M layers to GPU» и «n_ctx = …» свежие сборки печатают только с ним (лог растёт на ~3 КБ за ответ).
LLAMA_ARGS = ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-fitt", "256", "-fitc", str(CTX), "-lv", "4"]
OFFLOADED = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
N_CTX = re.compile(r"llama_context: n_ctx\s+=\s+(\d+)")

# повторный exec этого файла не теряет запущенную модель
current = globals().get("current")  # {"engine", "repo", "variant"} запущенной модели
server = globals().get("server")  # процесс llama-server
exl = globals().get("exl")  # ExLlamaV3: model, tok, gen, stop
placed = globals().get("placed")  # [на GPU, всего] слоёв у llama.cpp; у EXL3 None — она целиком в видеопамяти или не грузится
ctx = globals().get("ctx")  # контекст запущенной модели в токенах: у llama.cpp — какой он выбрал, у EXL3 — CTX
sampler = globals().get("sampler")  # поток, который пишет нагрузку VM в STATS

# Нагрузка VM для графиков на странице. Вызовы ядра идут по одному, и запрос нагрузки через ядро ждал бы конца
# ответа модели — как раз когда графики нужнее всего. Поэтому поток в ядре пишет её в файл, а server.py забирает
# файл через colab download: тот ходит мимо ядра.
STATS = "/content/vm_stats.json"
EVERY, KEPT = 2, 300  # точка раз в 2 с, 300 точек — последние 10 минут


def emit(kind, **kw):
    print("\x1e" + json.dumps({"kind": kind, **kw}, ensure_ascii=False), flush=True)


def cpu_times(text):
    """(простой, всего) из первой строки /proc/stat: процент CPU — доля непростоя между двумя замерами."""
    v = [int(x) for x in text.split("\n", 1)[0].split()[1:]]
    return v[3] + v[4], sum(v[:8])  # idle + iowait; guest уже входит в user


def ram_gb(text):
    """(занято, всего) ГБ из /proc/meminfo: занято — всё, кроме MemAvailable (кэш файлов отдаётся по первой просьбе)."""
    kb = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in text.splitlines() if ":" in line)}
    return round((kb["MemTotal"] - kb["MemAvailable"]) / 1e6, 2), round(kb["MemTotal"] / 1e6, 1)


def gpu_now():
    """(загрузка %, занято ГБ, всего ГБ) или None без GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        util, used, total = (int(x) for x in out.split("\n")[0].split(","))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return util, round(used * 1.048576e-3, 2), round(total * 1.048576e-3, 1)


def sample_forever():
    """Раз в EVERY с дописывает точку [время, CPU %, RAM ГБ, GPU %, видеопамять ГБ] в STATS, хранит KEPT последних."""
    points, was = [], (0, 0)
    while True:
        time.sleep(EVERY)
        try:  # поток не должен умереть с трейсбеком: тот попал бы в вывод чужого вызова ядра
            now = cpu_times(open("/proc/stat").read())
            idle, total = now[0] - was[0], now[1] - was[1]
            first, was = not was[1], now
            if first:  # первый замер — только точка отсчёта для процента CPU
                continue
            ram, ram_total = ram_gb(open("/proc/meminfo").read())
            gpu = gpu_now()
            points = (points + [[round(time.time(), 1), round(100 * (1 - idle / total), 1) if total else 0, ram,
                                 gpu and gpu[0], gpu and gpu[1]]])[-KEPT:]
            with open(STATS + ".tmp", "w") as f:  # через переименование: colab download не увидит файл недописанным
                json.dump({"ram_total": ram_total, "vram_total": gpu and gpu[2], "every": EVERY, "points": points}, f)
            os.replace(STATS + ".tmp", STATS)
        except Exception:  # noqa: BLE001, S112 — пропускаем точку, следующая через EVERY с
            continue


def watch():
    """Запускает поток нагрузки, если его ещё нет: он живёт в ядре, пока жива VM."""
    global sampler
    if not (sampler and sampler.is_alive()):
        sampler = threading.Thread(target=sample_forever, daemon=True, name="vm-stats")
        sampler.start()


def entry(name, **kw):
    """Точка входа ядра: любая ошибка становится кадром error с чистым текстом, а не трейсбеком в выводе."""
    try:
        watch()
        globals()[name](**kw)
    except BaseException as e:  # noqa: BLE001 — kernel.py превращает кадр в KernelError
        emit("error", text=str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}")


def wait(fn, stage, **progress):
    """fn() в потоке, раз в 2 с печатает этап: server.py показывает его, а colab exec не скучает от тишины."""
    box = {}

    def work():
        try:
            box["result"] = fn()
        except BaseException as e:  # noqa: BLE001 — пробрасываем ниже в основном потоке
            box["error"] = e

    th = threading.Thread(target=work)
    th.start()
    while th.is_alive():
        th.join(2)
        emit("stage", stage=stage, **{k: f() for k, f in progress.items()})
    if "error" in box:
        raise box["error"]
    return box.get("result")


def du(path):
    return sum(os.path.getsize(f) for f in glob.glob(f"{path}/**", recursive=True) if os.path.isfile(f) and not os.path.islink(f))


def vram():
    """Занятая видеопамять, ГБ."""
    out = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout
    return round(int(out.split()[0]) * 1.048576e-3, 1) if out.strip() else None


def install_llama():
    """Свежая сборка llama.cpp с CUDA (новые архитектуры моделей), при неудаче — LLAMA_PIN. Один раз за жизнь VM."""
    found = glob.glob("/content/llama-b*/llama-server")
    if found:
        return os.path.dirname(found[0])
    gh = "https://github.com/ggml-org/llama.cpp/releases/download"
    builds = []
    try:
        rels = json.load(urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10", timeout=20))
        builds = [(m[1], m[0]) for r in rels for a in r["assets"] if (m := re.fullmatch(r"llama-(b\d+)-bin-ubuntu-cuda-[\d.]+-x64\.tar\.gz", a["name"]))][:1]
    except OSError:
        pass  # GitHub API недоступен — сразу закреплённая сборка
    for tag, name in builds + [(LLAMA_PIN, f"llama-{LLAMA_PIN}-bin-ubuntu-cuda-12.8-x64.tar.gz")]:
        cmd = (f"curl -sfL {gh}/{tag}/{name} | tar -xz -C /content && "
               f"curl -sfL {gh}/{tag}/cudart-{name} | tar -xz -C /content/llama-{tag} --strip-components=1")
        if wait(lambda: subprocess.run(cmd, shell=True).returncode, f"Ставлю llama.cpp {tag}") == 0:
            return f"/content/llama-{tag}"
    raise RuntimeError("llama.cpp не скачалась с GitHub")


def install_exl3():
    """Колесо ExLlamaV3 под torch, CUDA и Python этой VM. Один раз за жизнь VM."""
    try:
        import exllamav3  # noqa: F401
        return
    except ImportError:
        pass
    import torch

    cu = "cu132" if torch.version.cuda.startswith("13") else "cu128"
    want = f"+{cu}.torch{torch.__version__.split('+')[0]}-cp{sys.version_info[0]}{sys.version_info[1]}-"
    rel = json.load(urllib.request.urlopen("https://api.github.com/repos/turboderp-org/exllamav3/releases/latest", timeout=20))
    whl = next((a["browser_download_url"] for a in rel["assets"] if want in a["name"] and a["name"].endswith("linux_x86_64.whl")), None)
    if not whl:
        raise RuntimeError(f"нет колеса ExLlamaV3 {rel['tag_name']} под torch {torch.__version__} и Python {sys.version_info[0]}.{sys.version_info[1]}")
    if wait(lambda: subprocess.run(f"pip install -q {whl}", shell=True).returncode, "Ставлю ExLlamaV3"):
        raise RuntimeError("pip не поставил ExLlamaV3")


def fetch(repo, variant, total):
    """Качает GGUF-файл или ветку EXL3 в кэш HF с прогрессом; возвращает путь к файлу или папке."""
    from huggingface_hub import constants, hf_hub_download, snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()  # прогресс печатаем сами, полосы tqdm только засоряют вывод
    constants.HF_HUB_DISABLE_XET = True  # Xet копит куски в своём кэше, и по размеру папки модели прогресс не виден
    folder = f"{CACHE}/models--{repo.replace('/', '--')}"
    base = du(folder)
    parts = re.sub(r"-00001-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", variant)
    if parts != variant:  # модель из частей: качаем все, llama-server сам найдёт их рядом с первой
        get = lambda: os.path.join(snapshot_download(repo, allow_patterns=[parts]), variant)  # noqa: E731
    elif variant.endswith(".gguf"):
        get = lambda: hf_hub_download(repo, variant)  # noqa: E731
    else:
        get = lambda: snapshot_download(repo, revision=variant)  # noqa: E731
    return wait(get, "Скачиваю", done=lambda: du(folder) - base, total=lambda: total)


def models():
    """Модели, скачанные на VM: GGUF-файлы и ветки EXL3 из кэша HF."""
    out = []
    for repo_dir in glob.glob(f"{CACHE}/models--*"):
        repo = os.path.basename(repo_dir)[8:].replace("--", "/", 1)
        branch = {open(f).read().strip(): os.path.relpath(f, f"{repo_dir}/refs") for f in glob.glob(f"{repo_dir}/refs/**", recursive=True) if os.path.isfile(f)}
        for snap in glob.glob(f"{repo_dir}/snapshots/*"):
            for f in glob.glob(f"{snap}/**/*.gguf", recursive=True):
                name = os.path.basename(f)
                if "mmproj" in name or re.match(r"mtp[-_]", name, re.I) or (re.search(r"-\d{5}-of-\d{5}\.gguf$", name) and "-00001-of-" not in name):
                    continue  # не модели и не первые части; у первой части размер — сумма всех
                size = sum(map(os.path.getsize, glob.glob(f.replace("-00001-of-", "-*-of-")))) if "-00001-of-" in name else os.path.getsize(f)
                out.append({"engine": "llama", "repo": repo, "variant": os.path.relpath(f, snap), "size": size})
            try:
                exl3 = json.load(open(f"{snap}/config.json"))["quantization_config"]["quant_method"] == "exl3"
                parts = set(json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"].values()) if os.path.exists(f"{snap}/model.safetensors.index.json") else {"model.safetensors"}
            except (OSError, KeyError, ValueError):
                continue
            if exl3 and all(os.path.exists(f"{snap}/{p}") for p in parts):  # недокачанную ветку не показываем
                size = sum(os.path.getsize(f"{snap}/{p}") for p in parts)
                out.append({"engine": "exl3", "repo": repo, "variant": branch.get(os.path.basename(snap), "main"), "size": size})
    return out


def state():
    """Что запущено и что скачано — одним кадром."""
    emit("state", running=current, models=models(), vram=vram(), layers=placed, ctx=ctx)


def placement(log):
    """[на GPU, всего] слоёв по последней строке offloaded в логе llama-server; None, если строки нет."""
    found = OFFLOADED.findall(log)
    return [int(n) for n in found[-1]] if found else None


def context(log):
    """Контекст, который выбрал llama-server, по последней строке n_ctx в логе; CTX, если строки нет."""
    found = N_CTX.findall(log)
    return int(found[-1]) if found else CTX


def unload():
    global server, exl, current, placed, ctx
    placed = ctx = None
    if server and server.poll() is None:
        server.terminate()
        server.wait()
    server = None
    if exl:
        exl["model"].unload()
        exl = None
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    current = None


def llama_log():
    """Хвост лога llama-server — в текст ошибки, чтобы было видно, почему он упал."""
    try:
        return open(LOG).read()[-1500:]
    except OSError:
        return ""


def llama_up():
    try:
        return json.load(urllib.request.urlopen(f"{URL}/health", timeout=2)).get("status") == "ok"
    except OSError:  # сервер ещё грузит модель (503) или не слушает порт
        return False


def load(engine, repo, variant, total=0):
    """Выгружает текущую модель, ставит движок, качает и запускает новую; прогресс — кадрами stage."""
    global server, exl, current, placed, ctx
    t = time.time()
    if current == {"engine": engine, "repo": repo, "variant": variant}:
        emit("ready", vram=vram(), secs=0)
        return
    unload()
    if engine == "llama":
        bin_dir = install_llama()
        path = fetch(repo, variant, total)
        # -ngl не задаём: llama.cpp сам кладёт на GPU сколько влезет, остальное — на CPU
        # CUDA_CACHE_MAXSIZE: ядра, скомпилированные под T4 при прогреве, остаются в кэше до конца жизни VM
        server = subprocess.Popen([f"{bin_dir}/llama-server", "-m", path, "--jinja", "--port", str(PORT), *LLAMA_ARGS],
                                  stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
                                  env={**os.environ, "CUDA_CACHE_MAXSIZE": str(4 << 30)})
        while not llama_up():
            if server.poll() is not None:
                raise RuntimeError("llama-server упал:\n" + llama_log())
            emit("stage", stage="Гружу в видеопамять")
            time.sleep(2)
        log = open(LOG).read()
        placed, ctx = placement(log), context(log)
        current = {"engine": engine, "repo": repo, "variant": variant}
    else:
        install_exl3()
        path = fetch(repo, variant, total)
        from exllamav3 import Cache, Config, Generator, Model, Tokenizer

        config = Config.from_directory(path)
        model = Model.from_config(config)
        cache = Cache(model, max_num_tokens=CTX)
        wait(model.load, "Гружу в видеопамять")
        tok = Tokenizer.from_config(config)
        exl = {"model": model, "tok": tok, "gen": Generator(model=model, cache=cache, tokenizer=tok), "stop": list(config.eos_token_id_list)}
        ctx = CTX
        current = {"engine": engine, "repo": repo, "variant": variant}
    # первый запрос ждёт JIT-компиляцию ядер под T4 — прогреваем, чтобы первое сообщение в чате не тормозило
    wait(lambda: chat([{"role": "user", "content": "Расскажи короткую историю про кота. " * 4}], max_tokens=16, quiet=True),
         "Прогреваю: компилирую ядра CUDA под T4, в первый раз до нескольких минут")
    emit("ready", vram=vram(), secs=round(time.time() - t))


def chat(messages, think=False, quiet=False, **params):
    """Отвечает на готовый список сообщений, печатая ответ потоком; рассуждения — в <think>…</think>.

    params: temperature, top_p, top_k, min_p, max_tokens, repeat_penalty, presence_penalty, frequency_penalty, seed, stop,
    tools и tool_choice в формате OpenAI. Бюджет ответа — весь остаток контекста; max_tokens его только сужает.
    Ответ печатается между кадрами begin и tps; в tps — скорость, причина остановки, вызовы инструментов (calls)
    и токены: prompt — сколько заняла переписка, made — сколько написала модель.
    Причина: stop — модель закончила сама, length — кончился max_tokens, context — кончился контекст, tool_calls — модель
    вызвала инструменты. Вызовы разбирает llama-server по шаблону модели (--jinja); у EXL3 инструментов нет.
    quiet=True — ничего не печатать (прогрев).
    """
    if not current:
        raise RuntimeError("модель не загружена")
    if not quiet:
        emit("begin")
    say = (lambda s: None) if quiet else (lambda s: print(s, end="", flush=True))
    if current["engine"] == "llama":
        r = chat_llama(messages, think, params, say)
    else:
        r = chat_exl3(messages, think, {k: v for k, v in params.items() if k not in ("tools", "tool_choice")}, say)
    asked = params.get("max_tokens")
    finish = "tool_calls" if r["calls"] else "stop" if not r["hit"] else "length" if asked and r["made"] >= asked else "context"
    if not quiet:
        print()
        emit("tps", tps=r["tps"], finish=finish, calls=r["calls"], prompt=r["prompt"], made=r["made"])
    return re.sub(r"<think>.*?</think>\s*", "", r["text"], flags=re.S)


def overflow():
    return RuntimeError(f"переписка не влезает в контекст {ctx or CTX} токенов — начни новый чат")


def chat_llama(messages, think, params, say):
    """→ {text, tps, hit — упёрлись ли в бюджет, made — сколько токенов написано, calls — вызовы инструментов,
    prompt — токенов в переписке}. Без max_tokens llama-server пишет до конца контекста.
    Вызовы приходят кусками по index — собираем целиком. Токены переписки — из timings: заново посчитанные (prompt_n)
    плюс взятые из кэша прошлого хода (cache_n)."""
    body = {
        "messages": messages,
        "stream": True,
        # Qwen понимает enable_thinking, gpt-oss — reasoning_effort; лишний ключ шаблон игнорирует
        "chat_template_kwargs": {"enable_thinking": think, "reasoning_effort": "medium" if think else "low"},
        **params,
    }
    req = urllib.request.Request(f"{URL}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if b"exceed_context_size" in e.read():
            raise overflow() from e
        raise RuntimeError(f"llama-server не ответил ({e}):\n" + llama_log()) from e
    except OSError as e:
        raise RuntimeError(f"llama-server не ответил ({e}):\n" + llama_log()) from e
    text, tps, thinking, hit, made, calls, prompt = "", None, False, False, 0, {}, 0
    for line in resp:
        if not line.startswith(b"data: {"):
            continue
        chunk = json.loads(line[6:])
        delta = chunk["choices"][0]["delta"] if chunk.get("choices") else {}
        if chunk.get("choices") and chunk["choices"][0].get("finish_reason"):
            hit = chunk["choices"][0]["finish_reason"] == "length"
        for part in delta.get("tool_calls") or []:
            call = calls.setdefault(part.get("index", len(calls)), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            call["id"] = part.get("id") or call["id"]
            for key in ("name", "arguments"):
                call["function"][key] += (part.get("function") or {}).get(key) or ""
        for kind, part in (("think", delta.get("reasoning_content")), ("text", delta.get("content"))):
            if not part:
                continue
            if (kind == "think") != thinking:  # llama-server отдаёт рассуждения отдельным полем — заворачиваем в теги
                part = ("<think>" if not thinking else "</think>") + part
                thinking = not thinking
            text += part
            say(part)
        if "timings" in chunk:
            tps = round(chunk["timings"]["predicted_per_second"], 1)
            made = chunk["timings"].get("predicted_n", 0)
            prompt = chunk["timings"].get("prompt_n", 0) + chunk["timings"].get("cache_n", 0)
    if thinking:
        text += "</think>"
        say("</think>")
    return {"text": text, "tps": tps, "hit": hit, "made": made, "calls": [calls[i] for i in sorted(calls)], "prompt": prompt}


def chat_exl3(messages, think, params, say):
    from exllamav3 import Job
    from exllamav3.generator.sampler import ComboSampler

    ids = exl["tok"].hf_chat_template(messages, add_generation_prompt=True, enable_thinking=think)
    room = CTX - ids.shape[-1]
    if room < 16:
        raise overflow()
    p = {"temperature": 0.8, "top_p": 0.95, "top_k": 40, "min_p": 0.05, **params}  # умолчания как у llama.cpp
    sampler = ComboSampler(temperature=p["temperature"], top_p=p["top_p"], top_k=p["top_k"] or 0, min_p=p["min_p"],
                           rep_p=p.get("repeat_penalty", 1.0), pres_p=p.get("presence_penalty", 0.0), freq_p=p.get("frequency_penalty", 0.0))
    stop = p.get("stop") or []
    job = Job(input_ids=ids, max_new_tokens=min(p.get("max_tokens") or room, room), sampler=sampler, seed=p.get("seed"),
              stop_conditions=exl["stop"] + ([stop] if isinstance(stop, str) else list(stop)))
    gen = exl["gen"]
    gen.enqueue(job)
    text, tps, hit, made = "", None, False, 0
    while gen.num_remaining_jobs():
        for res in gen.iterate():
            text += res.get("text", "")
            say(res.get("text", ""))
            if res.get("eos"):
                hit, made = res.get("eos_reason") == "max_new_tokens", res.get("new_tokens", 0)
                if res.get("time_generate"):
                    tps = round(res["new_tokens"] / res["time_generate"], 1)
    return {"text": text, "tps": tps, "hit": hit, "made": made, "calls": [], "prompt": ids.shape[-1]}
