from __future__ import annotations

import re


SENTENCE_RE = re.compile(r"(.+?[。！？!?](?:[」』）】\"]*)?)")


def normalize_tts_text(text: str) -> str:
    return " ".join(text.split()).strip()


class SentenceChunker:
    def __init__(self) -> None:
        self._buffer = ""

    def push(self, delta: str) -> list[str]:
        self._buffer += delta
        output: list[str] = []
        while True:
            match = SENTENCE_RE.match(self._buffer)
            if match is None:
                break
            sentence = normalize_tts_text(match.group(1))
            if sentence:
                output.append(sentence)
            self._buffer = self._buffer[match.end() :]
        return output

    def flush(self) -> str:
        remainder = normalize_tts_text(self._buffer)
        self._buffer = ""
        return remainder

