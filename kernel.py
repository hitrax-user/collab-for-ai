# «Ядро VM»: модель живёт в ядре IPython на Colab VM, server.py говорит с ней только через Kernel.
# Реализация — llm_chat.py: kernel.py шлёт его в ядро с каждым вызовом и разбирает служебные кадры из вывода.
import json
import threading
from typing import NamedTuple

from colab_run import HERE, run

SESSION = "llm"  # сессия Colab, в которой живёт VM с моделью
SOURCE = HERE / "llm_chat.py"
# Что из запроса передаём модели; остальное (n, logprobs, quiet…) молча отбрасываем
PARAMS = {"temperature", "top_p", "top_k", "min_p", "max_tokens", "stop", "seed", "presence_penalty", "frequency_penalty", "repeat_penalty",
          "tools", "tool_choice"}


class KernelError(Exception):
    """Ошибка в ядре VM или связи с ним; текст готов для показа человеку."""


GONE = "VM погасла или сменилась — запрос прерван"


class Reply(NamedTuple):
    """Ответ модели. finish: stop — закончила сама, length — кончился max_tokens, context — кончился контекст,
    tool_calls — вызвала инструменты. calls — вызовы в формате OpenAI tool_calls.
    prompt_tokens — токенов в переписке, completion_tokens — сколько написала модель (0 — старый llm_chat.py их не шлёт)."""
    text: str
    tps: float | None
    finish: str
    calls: list
    prompt_tokens: int
    completion_tokens: int


class Kernel:
    """Модель в ядре VM аккаунта acc. Вызовы выполняются по одному, остальные ждут очереди.

    transport(code, timeout) отдаёт вывод кода кусками; по умолчанию — colab exec в сессии SESSION.
    Любая ошибка — KernelError, в текст ответа она не попадает.
    Ядро привязано к одной VM: погасла она или сменилась — close(), и дальше говорить с новым Kernel.
    """

    def __init__(self, acc, transport=None):
        self._procs = set()  # процессы colab exec, которые сейчас идут: close() их убивает
        self.transport = transport or (lambda code, timeout: run(code, acc, SESSION, timeout, self._procs))
        self._turn = threading.Lock()  # ядро выполняет один exec за раз
        self._count = threading.Lock()
        self._calls = 0  # выполняется и ждёт очереди
        self.closed = False

    def close(self):
        """VM этого ядра больше нет: идущий вызов обрывается, ждущие очереди сразу получают KernelError(GONE).

        Без этого exec к погасшей VM висит до своего таймаута (до часа) и держит очередь.
        """
        self.closed = True
        for p in list(self._procs):
            p.kill()

    @property
    def busy(self):
        """Ядро занято: вызов выполняется или ждёт очереди."""
        return self._calls > 0

    def state(self):
        """{"running": {"engine", "repo", "variant"} или None, "models": скачанные на VM, "vram": ГБ или None,
        "layers": [на GPU, всего] слоёв у llama.cpp или None, "ctx": контекст запущенной модели в токенах или None}."""
        f = self._call("state", {}, 120, "state")
        # layers и ctx через get: ядро, поднятое старым llm_chat.py, этих полей не шлёт
        return {"layers": f.get("layers"), "ctx": f.get("ctx"), **{k: f[k] for k in ("running", "models", "vram")}}

    def load(self, engine, repo, variant, size=0, on_progress=None):
        """Ставит движок, качает и прогревает модель, выгрузив прежнюю. on_progress(этап, скачано, всего). Возвращает секунды."""
        def frame(f):
            if f["kind"] == "stage" and on_progress:
                on_progress(f["stage"], f.get("done", 0), f.get("total"))

        return self._call("load", {"engine": engine, "repo": repo, "variant": variant, "total": int(size)}, 3600, "ready", frame)["secs"]

    def chat(self, messages, params=None, think=False, on_text=None):
        """Ответ на список сообщений (Reply); on_text получает текст по мере генерации.

        Инструменты — в params["tools"]; сообщения идут в модель как есть (с tool_calls и role tool).
        Без max_tokens бюджет ответа — весь остаток контекста.
        Рассуждения — внутри текста в <think>…</think>. Из params доходят только PARAMS с непустым значением.
        """
        params = {k: v for k, v in (params or {}).items() if k in PARAMS and v is not None and v != []}
        # час: 8192 токена даже на 2,4 ток/с (27B с частью слоёв на CPU) — около 57 минут; длинный контекст бывает только у малых быстрых моделей
        f = self._call("chat", {"messages": messages, "think": bool(think), **params}, 3600, "tps", on_text=on_text)
        return Reply(f["text"].strip(), f["tps"], f.get("finish", "stop"), f.get("calls") or [], f.get("prompt", 0), f.get("made", 0))

    def _call(self, fn, kw, timeout, result, on_frame=None, on_text=None):
        code = SOURCE.read_text(encoding="utf-8") + f"\nentry({json.dumps(fn)}, **json.loads({json.dumps(json.dumps(kw, ensure_ascii=False), ensure_ascii=False)}));\n"
        with self._count:
            self._calls += 1
        try:
            with self._turn:
                if self.closed:
                    raise KernelError(GONE)
                try:
                    return self._read(self.transport(code, timeout), result, on_frame, on_text)
                except KernelError:
                    if self.closed:
                        raise KernelError(GONE) from None
                    raise
        finally:
            with self._count:
                self._calls -= 1

    def _read(self, chunks, result, on_frame, on_text):
        """Разбирает вывод: кадры \\x1e{json}\\n — отдельно, текст между кадрами begin и tps — в on_text."""
        buf, text, tail, inside, got, error = "", "", "", False, None, None

        def plain(s):
            nonlocal text, tail
            tail = (tail + s)[-1200:]
            if inside:
                text += s
                on_text and on_text(s)

        try:
            for chunk in chunks:
                buf += chunk
                while buf:
                    i = buf.find("\x1e")
                    if i:
                        plain(buf if i == -1 else buf[:i])
                        buf = "" if i == -1 else buf[i:]
                        continue
                    j = buf.find("\n")
                    if j == -1:
                        break  # кадр пришёл не целиком — ждём остаток
                    raw, buf = buf[1:j], buf[j + 1:]
                    try:
                        f = json.loads(raw)
                    except ValueError:
                        plain(raw + "\n")
                        continue
                    if f.get("kind") == "error":
                        inside, error = False, error or f["text"]  # дочитываем: exec сам доходит до конца
                    elif f.get("kind") == "begin":
                        inside = True
                    elif f.get("kind") == result:
                        inside, got = False, {**f, "text": text}
                    elif on_frame:
                        on_frame(f)
            plain(buf)
        except Exception as e:  # noqa: BLE001 — связь, CLI, таймаут: человеку нужен текст, а не тип
            raise KernelError(error or str(e) or type(e).__name__) from e
        if error:
            raise KernelError(error)
        if got is None:
            raise KernelError(tail.strip() or "VM не ответила")
        return got
