# Выполнение кода в ядре Colab VM через ./colab-as … exec. Транспорт для kernel.py; ANSI и HERE берут server.py и menu.
import codecs
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANSI = re.compile(r"\x1b\[[0-9;]*m")  # цвета IPython-трейсбеков
LOST = "RuntimeError: Connection was lost."


def run(code, acc, session, timeout=600, procs=None):
    """Выполняет код в ядре VM и отдаёт вывод кусками по мере прихода (stderr вперемешку со stdout).

    Связь оборвалась до первого вывода — повторяет один раз. Посреди вывода — ConnectionError без повтора:
    код уже мог выполниться наполовину. procs — множество, где живёт процесс colab exec, пока идёт вызов:
    через него вызов обрывают из другого потока.
    """
    return retry(lambda: execute(code, acc, session, timeout, procs))


def execute(code, acc, session, timeout, procs=None):
    """Один colab exec: куски вывода без ANSI-цветов. Генератор закрыли раньше конца — процесс убит."""
    p = subprocess.Popen(
        [str(HERE / "colab-as"), acc, "exec", "-s", session, "--timeout", str(timeout)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    procs is not None and procs.add(p)
    try:
        p.stdin.write(code.encode())
        p.stdin.close()
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while chunk := p.stdout.read1(4096):
            yield ANSI.sub("", dec.decode(chunk))
        p.wait()
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        procs is not None and procs.discard(p)


def retry(start):
    """Куски вывода start() без строки LOST; на LOST до первого вывода запускает start() ещё раз."""
    # ponytail: CLI иногда не успевает открыть websocket за 10 с, падает с LOST и не завершается —
    # убиваем процесс и повторяем; чаще повторять, если Colab начнёт сбоить регулярно
    for attempt in range(2):
        out, buf, said = start(), "", False
        for chunk in out:
            buf += chunk
            if LOST in buf:
                out.close()
                said = said or bool(buf.replace(LOST, "").strip())
                break
            cut = len(buf) - held(buf)
            if cut:
                said = said or bool(buf[:cut].strip())
                yield buf[:cut]
                buf = buf[cut:]
        else:
            if buf:
                yield buf
            return
        if said or attempt:
            raise ConnectionError("связь с VM оборвалась")


def held(buf):
    """Длина хвоста buf, который может оказаться началом LOST: его рано отдавать."""
    return next((n for n in range(min(len(LOST), len(buf)), 0, -1) if LOST.startswith(buf[-n:])), 0)
