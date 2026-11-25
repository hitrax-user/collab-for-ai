# Страница управления и OpenAI-совместимый API на http://localhost:8000; запускается из ./menu.
# С моделью на VM говорит только через Kernel (kernel.py, «Ядро VM»), VM включает и гасит через ./colab-as.
# С Colab наружу ничего не открывается. Состояние — в словаре S: страница опрашивает /api/state, долгие действия идут в фоновых потоках.
import json
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from colab_run import ANSI, HERE
from kernel import SESSION, Kernel, KernelError

PORT = 8000
ACCOUNTS = Path.home() / ".colab-accounts"
HF_TOKEN = Path.home() / ".cache/huggingface/token"
HF = "https://huggingface.co"
IDLE = 30 * 60  # авто-стоп VM после 30 минут без запросов
KEEP = Path.home() / ".colab-keep"  # галочка «Держать VM» переживает перезапуск сервера: файл есть — включена
RETRY = 5 * 60  # VM не включилась при «Держать VM» (Colab не дал T4) — следующая попытка через 5 минут
FAST, SLOW = 12.5e9, 16e9  # GGUF на T4 по замерам README: целиком в VRAM / часть слоёв на CPU / дальше не влезет
EXL3_MAX = 13.5e9  # ponytail: ExLlama не выносит слои на CPU; порог прикинут по 30B-A3B 3.0bpw (12,2 ГБ), уточнить замером
EMAIL = re.compile(r"Email:\s*(\S+@\S+)")  # строка из colab whoami
PART = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)")  # номер части у модели из нескольких файлов

S = {
    "acc": None,
    "vm": "off",  # off | checking | starting | on | stopping | lost
    "vm_msg": "",  # почему VM нет: T4 не дали, Colab закрыл, авто-стоп
    "vm_since": None,
    "colab_url": None,
    "job": None,  # загрузка модели: этап, прогресс, ошибка
    "running": None,  # запущенная модель
    "models": [],  # скачанные на VM
    "last": time.time(),  # последний запрос к модели — для авто-стопа
    "keep": KEEP.exists(),  # «Держать VM»: не гасить за простоем, а пропавшую — включить снова с той же моделью
    "want": False,  # VM должна быть: включили её мы или подхватили; кнопка «Погасить» и смена аккаунта снимают
    "revive": None,  # какую модель грузить в перезапущенную VM: последняя запущенная
    "retry": 0,  # когда «Держать VM» последний раз пробовала включить VM
    "stats": None,  # нагрузка VM для графиков: {"ram_total", "vram_total", "every", "points": [[время, CPU %, RAM, GPU %, видеопамять]]}
    "seen": 0,  # когда страница последний раз спрашивала состояние: без неё нагрузку не забираем
}
KERNEL = None  # ядро VM аккаунта S["acc"]; при смене аккаунта — новое

# index.html и llm_chat.py читаются с диска на каждый запрос, а этот код — один раз при запуске.
# Поменяли его — страница уже новая, а сервер старый: новые кнопки молча получают 404.
CODE = [HERE / f for f in ("server.py", "kernel.py", "colab_run.py")]
STAMP = [f.stat().st_mtime for f in CODE]


def stale():
    """Код сервера на диске новее запущенного — нужен перезапуск."""
    return [f.stat().st_mtime for f in CODE] != STAMP


def use(acc):
    """Новое ядро для VM аккаунта acc. Прежнее закрывается: его вызовы шли к VM, которой больше нет."""
    global KERNEL
    old, KERNEL = KERNEL, Kernel(acc)
    S["acc"] = acc
    old and old.close()


def busy():
    return bool(KERNEL and KERNEL.busy)


def colab(*args, timeout=300):
    p = subprocess.run([str(HERE / "colab-as"), S["acc"], *args], capture_output=True, text=True, timeout=timeout)
    return ANSI.sub("", p.stdout + p.stderr), p.returncode


def name(m):
    return Path(PART.sub("", m["variant"])).stem if m["engine"] == "llama" else f"{m['repo'].split('/')[1]} · {m['variant']}"


def mark(engine, size):
    if engine == "exl3":
        return "fast" if size <= EXL3_MAX else "no"
    return "fast" if size <= FAST else "slow" if size <= SLOW else "no"


# ---------- VM ----------

def bg(fn, *args):
    """Фоновое действие; если оно упало, страница покажет ошибку, а не вечное «Создаётся…»."""
    def work():
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001 — любая ошибка CLI или сети
            S["vm_msg"] = f"Ошибка: {e}"
            if (S["job"] or {}).get("active"):
                S["job"].update(active=False, error=str(e))
            if S["vm"] in ("checking", "starting"):
                S["vm"] = "off"

    threading.Thread(target=work, daemon=True).start()


def check_vm():
    """Есть ли VM у аккаунта: True/False, None — CLI не ответил внятно. colab status ядро не занимает."""
    out, _ = colab("status", "-s", SESSION, timeout=60)
    return True if "Hardware:" in out else False if "not found" in out else None


def refresh():
    """Что запущено и что скачано на VM."""
    st = KERNEL.state()
    S["models"] = st["models"]
    S["running"] = st["running"] and {**st["running"], "name": name(st["running"]), "vram": st["vram"], "layers": st["layers"], "ctx": st["ctx"]}
    if r := st["running"]:
        size = next((m["size"] for m in st["models"] if (m["engine"], m["repo"], m["variant"]) == (r["engine"], r["repo"], r["variant"])), 0)
        S["revive"] = {"engine": r["engine"], "repo": r["repo"], "variant": r["variant"], "size": size}


def colab_url():
    return next(iter(re.findall(r"https://\S+", colab("url", "-s", SESSION, timeout=60)[0])), None)


def attach():
    """Подхватывает VM аккаунта, если она уже есть: после перезапуска сервера или смены аккаунта."""
    acc = S["acc"]
    S.update(vm="checking", vm_msg="", running=None, models=[], job=None, colab_url=None, revive=None)
    alive = check_vm()
    if S["acc"] != acc:  # пока проверяли, переключились на другой аккаунт — его проверяет свой attach
        return
    S["want"] = bool(alive)  # VM прежнего аккаунта «Держать VM» на новом не включает
    if alive:
        S.update(vm="on", vm_since=time.time(), last=time.time(), colab_url=colab_url())
        refresh()
    else:
        S["vm"] = "off"


def switch(acc):
    """Смена аккаунта одним действием: VM прежнего аккаунта гасится, VM нового подхватывается, если есть."""
    if S["vm"] == "on":
        stop_vm()
    use(acc)
    attach()


def start_vm():
    S.update(vm="starting", vm_msg="", job=None)
    use(S["acc"])  # вызовы к прежней VM, если ещё висят, обрываем: новая их не дождётся
    out, rc = colab("new", "-s", SESSION, "--gpu", "T4")
    if rc or "READY" not in out:
        S.update(vm="off", vm_msg="Colab сейчас не даёт T4: кончилась квота или нет свободных. Попробуй позже или другой аккаунт."
                 if "400" in out else "VM не создалась: " + (out.strip()[-600:] or "CLI ничего не ответил"))
        return
    if HF_TOKEN.exists():
        colab("upload", "-s", SESSION, str(HF_TOKEN), "/content/.hf_token")
    S.update(vm="on", vm_since=time.time(), last=time.time(), running=None, models=[], colab_url=colab_url(), want=True)
    bg(warm)


def warm():
    """Первый вызов ядра на новой VM: он запускает поток нагрузки, и графики появляются, не дожидаясь загрузки модели.
    Сразу после «READY» VM может ещё не принимать код — тогда пробуем снова, а ошибку на страницу не выносим."""
    for _ in range(3):
        try:
            return refresh()
        except KernelError:
            time.sleep(10)


def stop_vm(msg=""):
    S.update(vm="stopping", want=False)
    colab("stop", "-s", SESSION, timeout=120)
    use(S["acc"])
    S.update(vm="off", vm_msg=msg, running=None, models=[], job=None, colab_url=None)


def revive():
    """«Держать VM»: VM пропала не по нашей воле — включаем снова и грузим последнюю модель."""
    model = S["revive"]
    start_vm()
    if S["vm"] != "on":
        S["vm_msg"] += f" Попробую снова через {RETRY // 60} мин."
    elif model and S["keep"]:
        load(model["engine"], model["repo"], model["variant"], model["size"])


def keep(on):
    """Галочка «Держать VM» со страницы; запоминается в файле KEEP."""
    S["keep"] = on
    KEEP.touch() if on else KEEP.unlink(missing_ok=True)


def tick():
    """Шаг monitor: жива ли VM, не пора ли её погасить за простоем, не пора ли включить снова."""
    if S["keep"] and S["want"] and S["vm"] in ("off", "lost") and time.time() - S["retry"] > RETRY:
        S["retry"] = time.time()
        bg(revive)
        return
    if S["vm"] != "on":
        return
    alive = check_vm()
    if alive is False:
        why = "Colab закрыл VM: простой или кончилась квота" + (". Включаю снова" if S["keep"] else "")
        S.update(vm="lost", vm_msg=why, running=None, models=[], job=None, colab_url=None, retry=0)
        use(S["acc"])
    elif alive and not S["keep"] and not busy() and not (S["job"] or {}).get("active") and time.time() - S["last"] > IDLE:
        stop_vm("VM погашена автоматически: 30 минут без запросов")


def monitor(stop):
    while not stop.wait(30):
        tick()


def fetch_stats():
    """Файл нагрузки с VM (его пишет поток в ядре, см. llm_chat.STATS). colab download ходит мимо ядра:
    графики идут и пока модель отвечает. Файла ещё нет — None."""
    tmp = Path(f"/tmp/colab-stats-{S['acc']}.json")
    tmp.unlink(missing_ok=True)
    try:
        colab("download", "-s", SESSION, "/content/vm_stats.json", str(tmp), timeout=60)
        return json.loads(tmp.read_text())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def stats_loop(stop):
    """Пока VM включена и страница открыта — свежая нагрузка раз в несколько секунд."""
    while not stop.wait(3):
        if S["vm"] == "on" and time.time() - S["seen"] < 30:
            acc, got = S["acc"], fetch_stats()
            if S["vm"] == "on" and S["acc"] == acc and got:
                S["stats"] = got
        elif S["vm"] != "on":
            S["stats"] = None


def load(engine, repo, variant, size):
    S["last"] = time.time()
    job = S["job"] = {"active": True, "name": name({"engine": engine, "repo": repo, "variant": variant}),
                      "stage": "Жду очередь", "since": time.time(), "done": 0, "total": size, "secs": None, "error": None}

    def progress(stage, done, total):
        if stage != job["stage"]:
            job["since"] = time.time()  # страница показывает, сколько идёт этап: видно, что не зависло
        job.update(stage=stage, done=done, total=total or job["total"])

    try:
        job.update(stage="Готово", secs=KERNEL.load(engine, repo, variant, size, progress))
    except KernelError as e:
        job["error"] = str(e)
    refresh()  # и после ошибки: прежняя модель уже могла выгрузиться
    job["active"] = False
    S["last"] = time.time()


def openai_finish(finish):
    """Причина остановки для OpenAI-клиентов: у них нет отдельного «кончился контекст», это тоже length."""
    return finish if finish in ("stop", "tool_calls") else "length"


def openai_message(m):
    """Сообщение клиента для модели: content — строкой, вызовы инструментов и ответы на них — как пришли."""
    return {"role": m["role"], "content": text_of(m.get("content")), **{k: m[k] for k in ("tool_calls", "tool_call_id", "name") if m.get(k)}}


def usage(r):
    """usage в формате OpenAI из ответа ядра: токены переписки и ответа по счёту самой модели."""
    return {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "total_tokens": r.prompt_tokens + r.completion_tokens}


def numbered(calls):
    """tool_calls для ответа: у каждого вызова id (llama-server может его не дать) и index для потока."""
    return [{**c, "index": i, "id": c.get("id") or f"call_{uuid.uuid4().hex[:24]}"} for i, c in enumerate(calls)]


def chat(messages, params, think, on_text):
    """Ответ модели на VM (kernel.Reply); текст отдаёт по мере генерации. Ошибка — KernelError."""
    S["last"] = time.time()
    try:
        return KERNEL.chat(messages, params, think, on_text)
    except KernelError:
        bg(refresh)  # ядро могло перезапуститься — страница не должна показывать модель, которой нет
        raise
    finally:
        S["last"] = time.time()


# ---------- Hugging Face ----------

def hf(path):
    headers = {"Authorization": f"Bearer {HF_TOKEN.read_text().strip()}"} if HF_TOKEN.exists() else {}
    return json.load(urllib.request.urlopen(urllib.request.Request(HF + path, headers=headers), timeout=20))


def parse(q):
    """Ссылка «huggingface.co/автор/модель[/tree|blob|resolve/ветка/путь]» или «автор/модель» → (репо, ветка, путь)."""
    q = re.sub(r"^(https?://)?(huggingface\.co|hf\.co)/", "", q.strip().split("?")[0].split("#")[0]).strip("/")
    parts = q.split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise ValueError("Нужна ссылка на модель или «автор/модель»")
    tail = len(parts) > 3 and parts[2] in ("tree", "blob", "resolve")
    return "/".join(parts[:2]), parts[3] if tail else None, urllib.parse.unquote("/".join(parts[4:])) if tail else ""


def group_gguf(files):
    """Варианты из дерева файлов репозитория. Модель из частей (-00001-of-00002) — один вариант с общим размером.
    mmproj (визуальный модуль) и mtp-*.gguf (MTP-голова для спекулятивного декодирования) — не модели, их не показываем."""
    out = {}
    for f in files:
        name = f["path"].rsplit("/", 1)[-1]
        if f["type"] != "file" or not name.endswith(".gguf") or "mmproj" in name or re.match(r"mtp[-_]", name, re.I):
            continue
        v = out.setdefault(PART.sub("", f["path"]), {"variant": f["path"], "size": 0})
        v["size"] += f["size"]
        if "-00001-of-" in name:  # грузится по первой части, остальные llama-server найдёт рядом
            v["variant"] = f["path"]
    return list(out.values())


def gguf_variants(repo):
    return group_gguf(hf(f"/api/models/{repo}/tree/main?recursive=true"))


def exl3_variants(repo, info):
    """Ветки EXL3 с размером весов; если веса лежат прямо в main — вариант один."""
    if any(s["rfilename"].endswith(".safetensors") for s in info.get("siblings", [])):
        branches = ["main"]
    else:
        branches = [b["name"] for b in hf(f"/api/models/{repo}/refs")["branches"] if b["name"] != "main"]

    def size(b):
        return sum(f.get("size", 0) for f in hf(f"/api/models/{repo}/tree/{urllib.parse.quote(b, safe='')}") if f["path"].endswith(".safetensors"))

    with ThreadPoolExecutor(8) as ex:
        return [{"variant": b, "size": s} for b, s in zip(branches, ex.map(size, branches)) if s]


def lookup(q, engine):
    """Варианты модели по ссылке: формат определяет движок; для оригинала ищем кванты под выбранный движок."""
    repo, rev, path = parse(q)
    try:
        info = hf(f"/api/models/{repo}")
    except urllib.error.HTTPError as e:
        raise ValueError(f"Не открылся {repo} на Hugging Face ({e.code}): опечатка или приватный репозиторий") from e
    tags = set(info.get("tags", []))
    note, alternatives = "", []
    if path.endswith(".gguf") or "gguf" in tags:
        engine, variants = "llama", gguf_variants(repo)
    elif "exl3" in tags:
        engine, variants = "exl3", exl3_variants(repo, info)
    else:  # оригинал, AWQ, GPTQ, EXL2…: ищем GGUF или EXL3 той же модели
        # у квантов база — в теге base_model:quantized:…; у оригинала base_model указывает на Base-модель, она не нужна
        base = next((t.split(":", 2)[2] for t in tags if t.startswith("base_model:quantized:")), repo)
        fmt = "gguf" if engine == "llama" else "exl3"
        found = [m["id"] for m in hf(f"/api/models?filter=base_model:quantized:{base}&filter={fmt}&sort=downloads&direction=-1&limit=5")]
        if not found:
            raise ValueError(f"Это не {fmt.upper()}, и {fmt.upper()}-версий {base} на HF не нашлось")
        alternatives = found
        # unsloth чинит шаблоны чата и делает Dynamic-кванты; у turboderp (автор ExLlamaV3) все bpw в одном репозитории
        repo = next((r for r in found if r.startswith("unsloth/" if fmt == "gguf" else "turboderp/")), found[0])
        rev, path = None, ""
        note = f"Это не {fmt.upper()} — взял {fmt.upper()}-версию {repo}"
        info = hf(f"/api/models/{repo}")
        variants = gguf_variants(repo) if fmt == "gguf" else exl3_variants(repo, info)
    if not variants:
        raise ValueError(f"В {repo} нет вариантов, которые умеет движок")
    for v in variants:
        v["mark"] = mark(engine, v["size"])
    variants.sort(key=lambda v: v["size"])
    want = path if engine == "llama" else rev
    fast = [v for v in variants if v["mark"] == "fast"]
    default = want if any(v["variant"] == want for v in variants) else (fast[-1] if fast else variants[0])["variant"]
    if info.get("gated"):
        note = (note + ". " if note else "") + f"Модель закрытая: прими лицензию на {HF}/{repo}, иначе скачивание упадёт"
    return {"engine": engine, "repo": repo, "variants": variants, "default": default, "alternatives": alternatives, "note": note}


# ---------- HTTP ----------

def accounts():
    out = []
    for d in sorted(ACCOUNTS.iterdir()) if ACCOUNTS.exists() else []:
        if d.is_dir() and not d.name.startswith("adding-"):  # adding- — вход ещё не закончен
            out.append({"name": d.name, "email": (d / "email").read_text().strip() if (d / "email").exists() else ""})
    return out


def adopt(tmp, email):
    """Папка входа adding-… получает имя по email. Неудачный вход или дубль — папка удаляется, ValueError."""
    same = email and next((a["name"] for a in accounts() if a["email"] == email), None)
    if not email or same:
        shutil.rmtree(ACCOUNTS / tmp, ignore_errors=True)
        raise ValueError(f"Этот аккаунт уже добавлен: {same}" if same else "Вход не удался — аккаунт не добавлен")
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", email.split("@")[0]).strip("-_") or "acc"
    name, n = base, 2
    while (ACCOUNTS / name).exists():
        name, n = f"{base}{n}", n + 1
    (ACCOUNTS / tmp).rename(ACCOUNTS / name)
    (ACCOUNTS / name / "email").write_text(email)
    return name


# Добавление аккаунта со страницы: colab whoami в новой папке печатает ссылку входа и ждёт код на stdin.
ADD = {"proc": None, "tmp": None, "url": None, "out": []}
GOOGLE_URL = re.compile(r"https://accounts\.google\.com/\S+")


def add_cancel():
    p, tmp = ADD["proc"], ADD["tmp"]
    ADD.update(proc=None, tmp=None, url=None, out=[])
    if p and p.poll() is None:
        p.kill()
        p.wait()
    if tmp:
        shutil.rmtree(ACCOUNTS / tmp, ignore_errors=True)


def add_start():
    """Запускает вход и возвращает ссылку Google; процесс ждёт код до 15 минут."""
    add_cancel()
    tmp = f"adding-{int(time.time())}"
    p = subprocess.Popen([str(HERE / "colab-as"), tmp, "whoami"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    out = []
    ADD.update(proc=p, tmp=tmp, out=out)
    threading.Thread(target=lambda: out.extend(ANSI.sub("", line) for line in p.stdout), daemon=True).start()
    timer = threading.Timer(15 * 60, lambda: ADD["proc"] is p and add_cancel())
    timer.daemon = True
    timer.start()
    deadline = time.time() + 30
    while time.time() < deadline and p.poll() is None:
        if m := GOOGLE_URL.search("".join(out)):
            ADD["url"] = m[0]
            return m[0]
        time.sleep(0.2)
    log = "".join(out).strip()[-400:]
    add_cancel()
    raise ValueError("colab не выдал ссылку для входа" + (f": {log}" if log else ""))


def add_finish(code):
    """Отдаёт код процессу входа; при успехе — имя нового аккаунта."""
    p, tmp, out = ADD["proc"], ADD["tmp"], ADD["out"]
    if not p or p.poll() is not None:
        add_cancel()
        raise ValueError("Вход устарел — нажми «Добавить аккаунт» ещё раз")
    try:
        p.stdin.write(code.strip() + "\n")
        p.stdin.close()
        p.wait(timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pass
    time.sleep(0.2)  # дочитать хвост вывода
    m = EMAIL.search("".join(out))
    ADD.update(tmp=None)  # папку дальше ведёт adopt
    add_cancel()
    if not m:
        shutil.rmtree(ACCOUNTS / tmp, ignore_errors=True)
        raise ValueError("Google не принял код. Нажми «Добавить аккаунт» ещё раз и вставь свежий код целиком")
    name = adopt(tmp, m[1])
    if S["vm"] in ("off", "lost") and not busy() and not (S["job"] or {}).get("active"):
        bg(switch, name)  # VM нет — сразу переходим на новый аккаунт, ради него и добавляли
    return name


def public_state():
    models = [{**m, "name": name(m), "mark": mark(m["engine"], m["size"])} for m in S["models"]]
    idle = max(0, int(IDLE - (time.time() - S["last"]))) if S["vm"] == "on" and not S["keep"] else None
    return {**{k: S[k] for k in ("acc", "vm", "vm_msg", "vm_since", "colab_url", "job", "running", "keep", "stats")},
            "reviving": S["keep"] and S["want"] and S["vm"] in ("off", "lost"),
            "busy": busy(), "stale": stale(), "models": models, "idle_left": idle, "now": time.time(), "accounts": accounts(), "adding": ADD["url"], "api": f"http://host.docker.internal:{PORT}/v1"}


def text_of(content):
    """OpenAI-клиенты шлют content строкой или списком частей [{"type": "text", "text": …}]."""
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # не засоряем терминал строкой на каждый опрос /api/state
        pass

    def reply(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def event(self, obj):
        try:
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        except OSError:  # вкладку закрыли — модель всё равно договорит, просто никому
            pass

    def body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        if url.path == "/":
            data = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif url.path == "/api/state":
            S["seen"] = time.time()
            self.reply(public_state())
        elif url.path == "/api/lookup":
            q = urllib.parse.parse_qs(url.query)
            try:
                self.reply(lookup(q.get("q", [""])[0], q.get("engine", ["llama"])[0]))
            except (ValueError, OSError) as e:
                self.reply({"error": str(e)}, 400)
        elif url.path.rstrip("/") == "/v1/models":
            label = f"Colab · {S['running']['name']}" if S["running"] else "Colab · модель не загружена"
            self.reply({"object": "list", "data": [{"id": "colab", "object": "model", "created": 0, "owned_by": "colab", "name": label}]})
        else:
            self.reply({"error": "not found"}, 404)

    def do_POST(self):
        path, body = self.path.rstrip("/"), self.body()
        if path == "/api/account":
            if body.get("name") not in [a["name"] for a in accounts()]:
                return self.reply({"error": "Нет такого аккаунта"}, 400)
            if body["name"] == S["acc"]:
                return self.reply({"ok": True})
            if S["vm"] in ("starting", "stopping") or (S["job"] or {}).get("active") or busy():
                return self.reply({"error": "Подожди: VM включается, гаснет или занята моделью"}, 409)
            bg(switch, body["name"])
        elif path == "/api/account/add":
            try:
                return self.reply({"url": add_start()})
            except (ValueError, OSError) as e:
                return self.reply({"error": str(e)}, 502)
        elif path == "/api/account/code":
            try:
                return self.reply({"name": add_finish(body.get("code", ""))})
            except ValueError as e:
                return self.reply({"error": str(e)}, 400)
        elif path == "/api/account/cancel":
            add_cancel()
        elif path == "/api/vm/start":
            if S["vm"] not in ("off", "lost"):
                return self.reply({"error": "VM уже есть"}, 409)
            bg(start_vm)
        elif path == "/api/vm/stop":
            if S["vm"] != "on" or (S["job"] or {}).get("active"):
                return self.reply({"error": "Сейчас нельзя: VM нет или идёт загрузка"}, 409)
            bg(stop_vm)
        elif path == "/api/keep":
            keep(bool(body.get("on")))
        elif path == "/api/load":
            if S["vm"] != "on" or (S["job"] or {}).get("active"):
                return self.reply({"error": "Сначала включи VM и дождись конца загрузки"}, 409)
            S["job"] = {"active": True, "name": name(body), "stage": "Жду очередь", "since": time.time(), "done": 0, "total": 0, "secs": None, "error": None}
            bg(load, body["engine"], body["repo"], body["variant"], body.get("size", 0))
        elif path == "/api/chat":
            if not S["running"]:
                return self.reply({"error": "Модель не загружена"}, 409)
            self.stream()
            try:
                r = chat(body["messages"], body.get("params"), body.get("think"), lambda t: self.event({"t": t}))
            except KernelError as e:
                return self.event({"error": str(e)})
            self.event({"done": True, "tps": r.tps, "finish": r.finish, "model": S["running"] and S["running"]["name"]})
            return
        elif path == "/v1/chat/completions":
            return self.completions(body)
        else:
            return self.reply({"error": "not found"}, 404)
        self.reply({"ok": True})

    def completions(self, body):
        if not S["running"]:
            return self.reply({"error": {"message": "Модель не загружена: загрузи её на http://localhost:8000"}}, 503)
        params = dict(body)  # лишнее (messages, n, logprobs…) отбросит Kernel.chat; tools и tool_choice дойдут до модели
        if params.get("max_tokens") is None:
            params["max_tokens"] = body.get("max_completion_tokens")
        think = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking"))
        messages = [openai_message(m) for m in body["messages"]]
        base = {"id": f"chatcmpl-{uuid.uuid4().hex}", "created": int(time.time()), "model": "colab"}
        if not body.get("stream"):
            try:
                r = chat(messages, params, think, None)
            except KernelError as e:
                return self.reply({"error": {"message": str(e)}}, 502)
            msg = {"role": "assistant", "content": r.text or (None if r.calls else "")}
            if r.calls:
                msg["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in numbered(r.calls)]
            return self.reply({**base, "object": "chat.completion", "usage": usage(r),
                               "choices": [{"index": 0, "message": msg, "finish_reason": openai_finish(r.finish)}]})
        self.stream()
        first = [True]

        def send(t):
            delta = {"role": "assistant", "content": t} if first[0] else {"content": t}
            first[0] = False
            self.event({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})

        try:
            r = chat(messages, params, think, send)
            if r.calls:  # вызовы — одним куском после текста: аргументы целиком, клиенту склеивать нечего
                delta = {"tool_calls": numbered(r.calls), **({"role": "assistant"} if first[0] else {})}
                self.event({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            # usage: по stream_options.include_usage — отдельным кадром без choices, как у OpenAI;
            # без него — в последнем кадре, как у llama-server: клиенты берут его оттуда
            asked = (body.get("stream_options") or {}).get("include_usage")
            last = {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": openai_finish(r.finish)}]}
            self.event(last if asked else {**last, "usage": usage(r)})
            if asked:
                self.event({**base, "object": "chat.completion.chunk", "choices": [], "usage": usage(r)})
        except KernelError as e:
            self.event({"error": {"message": str(e)}})
        try:
            self.wfile.write(b"data: [DONE]\n\n")
        except OSError:
            pass


def serve(acc):
    """Сервер для аккаунта acc до Ctrl-C; на выходе спрашивает, гасить ли VM."""
    use(acc)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        print(f"Порт {PORT} занят: закрой старый сервер или прокси и попробуй снова")
        return
    stop = threading.Event()
    bg(attach)
    threading.Thread(target=monitor, args=(stop,), daemon=True).start()
    threading.Thread(target=stats_loop, args=(stop,), daemon=True).start()
    url = f"http://localhost:{PORT}"
    print(f"\nСтраница: {url}\nAPI для Open WebUI: http://host.docker.internal:{PORT}/v1\nCtrl-C — выйти в меню\n")
    try:
        subprocess.Popen(["explorer.exe", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass  # не WSL — адрес выше, открой сам
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()
    if S["vm"] in ("on", "starting", "checking"):
        if input(f"\nПогасить VM аккаунта {S['acc']}? [Д/н] ").strip().lower() not in ("н", "n", "нет", "no"):
            stop_vm()
            print("VM погашена")
