from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


def synthesize_with_sarashina(
    *,
    reference_audio: Path,
    reference_text: str,
    text: str,
    output_path: Path,
    model_name: str,
) -> Path:
    try:
        from sarashina_tts.generate.generate import SarashinaTTSGenerator
    except ImportError as exc:
        raise RuntimeError(
            "sarashina2.2-tts is not installed. Install the official package from "
            "https://github.com/sbintuitions/sarashina2.2-tts, then run this command again."
        ) from exc

    generator = SarashinaTTSGenerator(model_id=model_name)
    audio_prompt_tokens = generator._extract_audio_prompt_tokens(
        audio_prompt_path=str(reference_audio)
    )
    flow_embedding = generator._extract_zero_shot_embedding(
        audio_prompt_path=str(reference_audio)
    )
    audio_prompt_feat = generator._extract_audio_prompt_feat(
        audio_prompt_path=str(reference_audio)
    )
    wavs = generator.generate(
        [text],
        flow_embedding=flow_embedding,
        audio_prompt_text=reference_text,
        audio_prompt_tokens=audio_prompt_tokens,
        audio_prompt_feat=audio_prompt_feat,
        audio_prompt_path=str(reference_audio),
        flow_embedding_only=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmpdir:
        tmp_output = Path(tmpdir)
        generator.save_audios(wavs, output_dir=str(tmp_output))
        generated_files = sorted(tmp_output.glob("*.wav"))
        if not generated_files:
            raise RuntimeError("sarashina2.2-tts did not create a wav file")
        shutil.move(str(generated_files[0]), output_path)

    return output_path
