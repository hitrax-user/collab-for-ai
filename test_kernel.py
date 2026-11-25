# Проверка «Ядра VM» через его интерфейс: python3 test_kernel.py (без сети, VM и GPU).
# Вместо colab exec — настоящий llm_chat.py в постоянном словаре-ядре, вместо llama-server — фейк на 127.0.0.1:8181.
# load() с движками здесь не проверить: ему нужны GPU и HF; его кадры проверяются отдельно, по записанному выводу.
import contextlib
import io
import json
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from colab_run import retry
from kernel import Kernel, KernelError


class Pipe(io.TextIOBase):
    def __init__(self, q):
        self.q = q

    def write(self, s):
        self.q.put(s)
        return len(s)


def vm(ns):
    """Транспорт-ядро: код выполняется в ns, как в IPython на VM; вывод идёт кусками по мере печати."""
    def transport(code, timeout):
        q = queue.Queue()

        def work():
            with contextlib.redirect_stdout(Pipe(q)), contextlib.redirect_stderr(Pipe(q)):
                try:
                    exec(code, ns)
                except BaseException:  # noqa: BLE001 — IPython тоже печатает трейсбек и живёт дальше
                    traceback.print_exc()
            q.put(None)

        threading.Thread(target=work).start()
        while (s := q.get()) is not None:
            yield s

    return transport


def script(*chunks, gate=None):
    """Транспорт, который отдаёт заранее записанный вывод; gate — ждать события перед выводом."""
    def transport(code, timeout):
        gate and gate.wait(5)
        for c in chunks:
            if isinstance(c, BaseException):
                raise c
            yield c

    return transport


SEEN = []  # тела запросов к фейковому llama-server


class Llama(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        said = body["messages"][-1]["content"]
        if said == "огромно":  # переписка больше контекста — так отвечает llama-server
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": {"code": 400, "type": "exceed_context_size_error", "message": "the request exceeds the available context size"}}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if said == "найди":  # вызов инструмента: llama-server шлёт его кусками по index, аргументы — по частям
            for chunk in ({"choices": [{"delta": {"content": "Ищу."}}]},
                          {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q": '}}]}}]},
                          {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"вал"}'}}]}}]},
                          {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "timings": {"predicted_per_second": 5.0, "predicted_n": 9}}):
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        # «длинно» — модель упирается в бюджет: в max_tokens, если он задан, иначе в контекст
        finish, made = ("length", body.get("max_tokens", 8000)) if said == "длинно" else ("stop", 3)
        for chunk in ({"choices": [{"delta": {"reasoning_content": "хм"}}]}, {"choices": [{"delta": {"content": "При"}}]},
                      {"choices": [{"delta": {"content": "вет"}}]},
                      {"choices": [{"delta": {}, "finish_reason": finish}], "timings": {"predicted_per_second": 42.42, "predicted_n": made, "prompt_n": 20, "cache_n": 5}}):
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def raises(fn, *args):
    try:
        fn(*args)
    except KernelError as e:
        return str(e)
    raise AssertionError("ждали KernelError")


llama = ThreadingHTTPServer(("127.0.0.1", 8181), Llama)
threading.Thread(target=llama.serve_forever, daemon=True).start()
said = []

# свежее ядро: ничего не запущено и не скачано
ns = {}
k = Kernel("test", vm(ns))
st = k.state()
assert st["running"] is None and st["models"] == [] and st["layers"] is None, st

# раскладка слоёв — из лога llama-server: последняя строка offloaded (авторазкладка может печатать и примерки)
log = """load_tensors: offloaded 40/65 layers to GPU
load_tensors:        CUDA0 model buffer size =  8694.30 MiB
0.04.364 I load_tensors: offloaded 61/65 layers to GPU
llama_kv_cache:      CUDA0 KV buffer size =   136.00 MiB
"""
assert ns["placement"](log) == [61, 65]
assert ns["placement"]("main: server is listening") is None
ns["placed"] = [61, 65]
assert k.state()["layers"] == [61, 65]
ns["placed"] = None

# контекст выбирает llama.cpp (--fit): берём последнюю строку n_ctx, n_ctx_seq — не она; строки нет — наименьший 8192
log = """llama_params_fit_impl: context size reduced from 262144 to 65536
llama_context: n_ctx         = 65536
llama_context: n_ctx_seq     = 65536
"""
assert ns["context"](log) == 65536 and ns["context"]("main: server is listening") == 8192
ns["ctx"] = 65536
assert k.state()["ctx"] == 65536
assert raises(Kernel("test", vm({**ns, "current": {"engine": "llama"}})).chat, [{"role": "user", "content": "огромно"}]).startswith("переписка не влезает в контекст 65536")
ns["ctx"] = None

# нагрузка VM: CPU — из /proc/stat (простой = idle + iowait), RAM — всё, кроме MemAvailable; поток нагрузки запущен первым же вызовом
assert ns["cpu_times"]("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 2 3\n") == (850, 1000)
assert ns["ram_gb"]("MemTotal:       12982012 kB\nMemFree: 1 kB\nMemAvailable:    9782012 kB\n") == (3.2, 13.0)
assert ns["sampler"].is_alive() and ns["sampler"].name == "vm-stats"

# без модели — KernelError с чистым текстом, в ответ ничего не утекает
assert raises(k.chat, [{"role": "user", "content": "hi"}], None, False, said.append) == "модель не загружена"
assert said == []

# модель в ядре: ответ потоком, рассуждения в <think>, скорость из timings; лишние параметры не доходят до модели
ns["current"] = {"engine": "llama", "repo": "u/r", "variant": "m.gguf"}
got = k.chat([{"role": "user", "content": "привет"}], {"temperature": 0.3, "quiet": True, "tools": [], "top_k": None}, True, said.append)
# токены переписки — посчитанные заново плюс взятые из кэша прошлого хода
assert got == ("<think>хм</think>Привет", 42.4, "stop", [], 25, 3), got
assert got.prompt_tokens == 25 and got.completion_tokens == 3
assert "".join(said).strip() == got[0]
body = SEEN[-1]
assert body["temperature"] == 0.3 and "quiet" not in body and "tools" not in body and "top_k" not in body, body
assert body["chat_template_kwargs"]["enable_thinking"] is True and body["messages"][0]["content"] == "привет"
# без max_tokens от клиента бюджет — весь остаток контекста, лимит модели не шлём
assert "max_tokens" not in body, body

# причина остановки: упёрлись в max_tokens клиента — length, в контекст — context
long = [{"role": "user", "content": "длинно"}]
assert k.chat(long, {"max_tokens": 64})[2] == "length"
assert SEEN[-1]["max_tokens"] == 64
assert k.chat(long)[2] == "context"

# инструменты: tools и tool_choice доходят до llama-server, вызов собирается из кусков, причина — tool_calls
tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
call = {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q": "вал"}'}}
got = k.chat([{"role": "user", "content": "найди"}], {"tools": tools, "tool_choice": "required"})
assert got == ("Ищу.", 5.0, "tool_calls", [call], 0, 9), got
assert SEEN[-1]["tools"] == tools and SEEN[-1]["tool_choice"] == "required"
# ход с результатом инструмента: tool_calls и role tool идут в модель как есть
turn = [{"role": "user", "content": "найди"}, {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "S1: снять муфту"}, {"role": "user", "content": "привет"}]
assert k.chat(turn, {"tools": tools})[2] == "stop"
assert SEEN[-1]["messages"][1:3] == turn[1:3], SEEN[-1]["messages"]

# переписка больше контекста — понятный текст, а не сырой ответ llama-server
assert raises(k.chat, [{"role": "user", "content": "огромно"}]) == "переписка не влезает в контекст 8192 токенов — начни новый чат"

# модель переживает повторную отправку исходника, а новое ядро (перезапуск VM) её теряет
assert k.state()["running"] == ns["current"]
assert raises(Kernel("test", vm({})).chat, [{"role": "user", "content": "hi"}]) == "модель не загружена"

# кадры, разрезанные как попало: текст только между begin и tps
out = 'шум\n\x1e{"kind": "begin"}\nПри\nвет\n\x1e{"kind": "tps", "tps": 7.5, "finish": "length"}\nхвост'
said.clear()
assert Kernel("test", script(*out)).chat([], on_text=said.append) == ("При\nвет", 7.5, "length", [], 0, 0)
assert "".join(said) == "При\nвет\n"

# кадр error — KernelError с его текстом; вывод до begin в ответ не попадает
said.clear()
out = ('Traceback…\n\x1e{"kind": "begin"}\nПр', '\x1e{"kind": "error", "text": "llama-server не ответил:\\nCUDA OOM"}\n')
assert raises(Kernel("test", script(*out)).chat, [], None, False, said.append) == "llama-server не ответил:\nCUDA OOM"
assert said == ["Пр"]

# вывод без кадра-результата (ядро упало, таймаут CLI) — KernelError с хвостом вывода, ничего не стримится
said.clear()
assert "NameError" in raises(Kernel("test", script("Traceback\n", "NameError: x\n")).chat, [], None, False, said.append)
assert said == [] and raises(Kernel("test", script()).state) == "VM не ответила"

# ошибка транспорта — тоже KernelError
assert raises(Kernel("test", script('\x1e{"kind": "begin"}\n', ConnectionError("связь с VM оборвалась"))).chat, []) == "связь с VM оборвалась"

# load: этапы — в on_progress, результат — секунды
steps = []
out = '\x1e{"kind": "stage", "stage": "Скачиваю", "done": 5, "total": 10}\n\x1e{"kind": "stage", "stage": "Прогреваю"}\n\x1e{"kind": "ready", "vram": 3.1, "secs": 42}\n'
assert Kernel("test", script(out)).load("llama", "u/r", "m.gguf", 10, lambda *a: steps.append(a)) == 42
assert steps == [("Скачиваю", 5, 10), ("Прогреваю", 0, None)], steps

# busy — пока вызов выполняется или ждёт очереди
gate = threading.Event()
k = Kernel("test", script('\x1e{"kind": "state", "running": null, "models": [], "vram": null, "layers": null}\n', gate=gate))
th = threading.Thread(target=k.state)
th.start()
for _ in range(200):  # поток ещё мог не войти в вызов
    if k.busy:
        break
    time.sleep(0.01)
assert k.busy and th.is_alive()
gate.set()
th.join(5)
assert not k.busy

# close(): VM погасла — идущий вызов кончается GONE, ждущий очереди даже не запускается
gate, ran, errs = threading.Event(), [], []


def hung(code, timeout):  # exec к погасшей VM: молчит, пока его не оборвут
    ran.append(1)
    gate.wait(5)
    yield "Traceback\n"


k = Kernel("test", hung)


def call():
    try:
        k.state()
    except KernelError as e:
        errs.append(str(e))


ths = [threading.Thread(target=call) for _ in range(2)]
for th in ths:
    th.start()
for _ in range(200):
    if k._calls == 2 and ran:
        break
    time.sleep(0.01)
k.close()
gate.set()
for th in ths:
    th.join(5)
assert errs == ["VM погасла или сменилась — запрос прерван"] * 2 and ran == [1], (errs, ran)
assert not k.busy

# транспорт colab exec: LOST до вывода — повтор, LOST посреди вывода — ошибка без повтора
LOST = "RuntimeError: Connection was lost."


def attempts(*runs):
    runs, n = list(runs), []

    def start():
        n.append(1)
        yield from runs.pop(0)

    return start, n


start, n = attempts(["Runtime", "Error: Connection was lost."], ["ok"])
assert list(retry(start)) == ["ok"] and len(n) == 2  # начало LOST не ушло наружу
start, n = attempts(["частичный ответ", LOST], ["ok"])
got = []
try:
    got.extend(retry(start))
    raise AssertionError("ждали ConnectionError")
except ConnectionError:
    assert got == ["частичный ответ"] and len(n) == 1
start, n = attempts([LOST], [LOST])
try:
    list(retry(start))
    raise AssertionError("ждали ConnectionError")
except ConnectionError:
    assert len(n) == 2
start, _ = attempts(["abc R", "ight"])
assert "".join(retry(start)) == "abc Right"  # хвост, похожий на начало LOST, придержан и отдан
llama.shutdown()
print("ok")
