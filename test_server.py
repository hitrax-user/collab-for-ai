# Проверка server.py: разбор ссылок и вариантов моделей, OpenAI API на записанном выводе ядра. python3 test_server.py (без сети и VM).
import json
import os
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import server
from kernel import Kernel, Reply
from server import HERE, group_gguf, mark, numbered, openai_finish, openai_message, parse, stale, usage

assert parse("https://huggingface.co/unsloth/Qwen3-8B-GGUF") == ("unsloth/Qwen3-8B-GGUF", None, "")
assert parse("Qwen/Qwen3-8B") == ("Qwen/Qwen3-8B", None, "")
assert parse("https://huggingface.co/u/r/blob/main/sub/x%20y.gguf?download=true") == ("u/r", "main", "sub/x y.gguf")
assert parse("hf.co/turboderp/Qwen3-30B-A3B-exl3/tree/3.0bpw") == ("turboderp/Qwen3-30B-A3B-exl3", "3.0bpw", "")
for bad in ("", "Qwen", "https://huggingface.co/"):
    try:
        parse(bad)
        raise AssertionError(bad)
    except ValueError:
        pass

assert [mark("llama", s) for s in (12e9, 13e9, 17e9)] == ["fast", "slow", "no"]
assert [mark("exl3", s) for s in (12.2e9, 14e9)] == ["fast", "no"]

# части одной модели — один вариант с общим размером; MTP-головы и mmproj — не модели, а noMTP — модель
files = [{"type": "file", "path": p, "size": n} for p, n in (
    ("Q2/m-Q2-00002-of-00002.gguf", 30), ("Q2/m-Q2-00001-of-00002.gguf", 40), ("m-Q4_K_M.gguf", 5),
    ("mtp-m-Q8_0.gguf", 4), ("mmproj-m-BF16.gguf", 1), ("m-noMTP-IQ2_M.gguf", 3), ("README.md", 1))]
assert sorted((v["variant"], v["size"]) for v in group_gguf(files)) == [
    ("Q2/m-Q2-00001-of-00002.gguf", 70), ("m-Q4_K_M.gguf", 5), ("m-noMTP-IQ2_M.gguf", 3)]

# код сервера поменяли после запуска — страница просит перезапустить, а не молча получает 404 на новые кнопки
assert not stale()
f = HERE / "kernel.py"
was = f.stat()
os.utime(f, (was.st_atime, was.st_mtime + 10))
try:
    assert stale()
finally:
    os.utime(f, (was.st_atime, was.st_mtime))
assert not stale()

# инструменты в OpenAI API: вызовы и ответы на них доходят до модели, у вызовов в ответе есть id и index
call = {"id": "", "type": "function", "function": {"name": "search", "arguments": "{}"}}
assert openai_message({"role": "assistant", "content": None, "tool_calls": [call]}) == {"role": "assistant", "content": "", "tool_calls": [call]}
assert openai_message({"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "S1"}]}) == {"role": "tool", "content": "S1", "tool_call_id": "c1"}
assert openai_message({"role": "user", "content": "hi", "tool_calls": []}) == {"role": "user", "content": "hi"}
got = numbered([call, {**call, "id": "c2"}])
assert [c["index"] for c in got] == [0, 1] and got[0]["id"].startswith("call_") and got[1]["id"] == "c2"
assert usage(Reply("x", 1.0, "stop", [], 25, 3)) == {"prompt_tokens": 25, "completion_tokens": 3, "total_tokens": 28}
assert [openai_finish(f) for f in ("stop", "tool_calls", "length", "context")] == ["stop", "tool_calls", "length", "length"]

# /v1/chat/completions целиком, с ядром на записанном выводе: текст, tool_calls и usage — со стримингом и без
server.S["running"] = {"engine": "llama", "repo": "u/r", "variant": "m.gguf", "name": "m"}
frames = '\x1e{"kind": "begin"}\nИщу.\n\x1e{"kind": "tps", "tps": 5.0, "finish": "tool_calls", "prompt": 40, "made": 9, ' \
         '"calls": [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]}\n'
server.KERNEL = Kernel("test", lambda code, timeout: iter([frames]))
httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req).read().decode()


ask = {"messages": [{"role": "user", "content": "найди"}], "tools": [{"type": "function", "function": {"name": "search"}}]}
want = {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49}
got = json.loads(post(ask))
assert got["usage"] == want and got["choices"][0]["finish_reason"] == "tool_calls", got
assert got["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "search"
events = [json.loads(line[6:]) for line in post({**ask, "stream": True}).splitlines() if line.startswith("data: {")]
assert events[-1]["usage"] == want and events[-1]["choices"][0]["finish_reason"] == "tool_calls", events[-1]
assert any(e["choices"][0]["delta"].get("tool_calls") for e in events[:-1])
events = [json.loads(line[6:]) for line in post({**ask, "stream": True, "stream_options": {"include_usage": True}}).splitlines() if line.startswith("data: {")]
assert events[-1] == {**events[-1], "choices": [], "usage": want} and "usage" not in events[-2], events[-2:]

# «Держать VM»: галочка со страницы, без неё — авто-стоп за простоем, с ней — перезапуск пропавшей VM
server.KEEP = Path(tempfile.mkdtemp()) / "keep"
req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/api/keep", b'{"on": true}', {"Content-Type": "application/json"})
assert json.load(urllib.request.urlopen(req)) == {"ok": True} and server.S["keep"] and server.KEEP.exists()
httpd.shutdown()

did = []
server.check_vm = lambda: did.append("check") or alive[0]
server.bg = lambda fn, *a: did.append(fn.__name__)
server.stop_vm = lambda msg="": did.append("stop")
server.use = lambda acc: None
server.busy = lambda: False
alive = [True]
server.S.update(vm="on", want=True, job=None, last=0)
server.tick()
assert did == ["check"], did  # простой больше 30 минут, но галочка стоит — не гасим
server.keep(False)
assert not server.KEEP.exists()
did.clear(); server.tick()
assert did == ["check", "stop"], did
server.keep(True)
alive[0] = False
did.clear(); server.tick()
assert server.S["vm"] == "lost" and "Включаю снова" in server.S["vm_msg"], server.S
did.clear(); server.tick(); server.tick()
assert did == ["revive"], did  # перезапуск один, следующая попытка — через RETRY
server.S.update(vm="off", want=False, retry=0)
did.clear(); server.tick()
assert did == [], did  # погасили кнопкой — не включаем

# нагрузка VM: файл с VM через colab download; файла ещё нет — None, а не ошибка
server.colab = lambda *a, timeout=0: (Path(a[-1]).write_text('{"every": 2, "points": [[1, 2, 3, 4, 5]]}'), ("", 0))[1]
assert server.fetch_stats()["points"] == [[1, 2, 3, 4, 5]]
server.colab = lambda *a, timeout=0: ("File not found", 1)
assert server.fetch_stats() is None
print("ok")
