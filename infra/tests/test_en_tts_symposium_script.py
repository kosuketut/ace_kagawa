"""The English TTS server must prewarm exactly the symposium sentences that tokkio_llm sends on Start."""

from __future__ import annotations

import importlib.util
import re
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EN_TTS_DIR = ROOT / "en-tts"
STUBBED_MODULES = (
    "loguru",
    "nvidia_pipecat",
    "nvidia_pipecat.services",
    "nvidia_pipecat.services.nvidia_llm",
    "pipecat",
    "pipecat.frames",
    "pipecat.frames.frames",
    "pipecat.processors",
    "pipecat.processors.aggregators",
    "pipecat.processors.aggregators.openai_llm_context",
    "pipecat.services",
    "pipecat.services.openai",
    "pipecat.services.openai.llm",
)


class _StubModule(types.ModuleType):
    """Resolves any imported name to a placeholder class, so tokkio_llm loads without pipecat."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        placeholder = type(name, (), {"__init__": lambda self, *args, **kwargs: None})
        setattr(self, name, placeholder)
        return placeholder


class _Logger:
    def __getattr__(self, _name: str):
        return lambda *_args, **_kwargs: None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_tokkio_llm_with_stubs():
    saved = {name: sys.modules.get(name) for name in STUBBED_MODULES}
    try:
        for name in STUBBED_MODULES:
            module = _StubModule(name)
            module.__path__ = []
            sys.modules[name] = module
        sys.modules["loguru"].logger = _Logger()
        return load_module("tokkio_llm_symposium_script_test", ROOT / "infra" / "tokkio" / "tokkio_llm.py")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


tokkio_llm = load_tokkio_llm_with_stubs()
# en-tts/ is optional in a checkout; load the server only when it is there.
HAS_EN_TTS = EN_TTS_DIR.is_dir()
indextts_server = (
    load_module("indextts_tts_server_symposium_test", EN_TTS_DIR / "indextts_tts_server.py") if HAS_EN_TTS else None
)


@unittest.skipUnless(HAS_EN_TTS, "en-tts/ not present")
class EnglishSymposiumScriptTests(unittest.TestCase):
    def test_indextts_prewarms_the_script_tokkio_llm_speaks(self) -> None:
        self.assertEqual(
            indextts_server.sentence_texts(indextts_server.DEFAULT_PREWARM_TEXT),
            tokkio_llm.SYMPOSIUM_OPENING_SCRIPT,
        )

    def test_deployed_unit_prewarms_the_checked_script_file(self) -> None:
        unit = (EN_TTS_DIR / "ace-indextts-tts.service").read_text(encoding="utf-8")
        match = re.search(r"^Environment=INDEXTTS_TTS_PREWARM_TEXT=(.+)$", unit, flags=re.MULTILINE)

        self.assertIsNotNone(match)
        self.assertEqual(Path(match.group(1)).resolve(), indextts_server.DEFAULT_PREWARM_TEXT.resolve())


if __name__ == "__main__":
    unittest.main()
