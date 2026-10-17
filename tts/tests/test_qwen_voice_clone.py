import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import qwen_voice_clone


class QwenVoiceCloneTests(unittest.TestCase):
    def test_parser_has_required_inputs_and_isolated_defaults(self):
        args = qwen_voice_clone._build_parser().parse_args(
            [
                "--reference-audio",
                "reference.wav",
                "--reference-text",
                "参照音声です",
                "--text",
                "生成音声です",
                "--output",
                "output.wav",
            ]
        )

        self.assertEqual(args.reference_audio, Path("reference.wav"))
        self.assertEqual(args.model, qwen_voice_clone.DEFAULT_MODEL)
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.attn_implementation, "sdpa")

    def test_run_uses_transcribed_voice_clone_and_reports_metadata(self):
        model = Mock()
        audio = [0.0] * 24000
        model.generate_voice_clone.return_value = ([audio], 24000)
        output = Path("/tmp/qwen-clone-test.wav")

        with patch.object(qwen_voice_clone, "_load_model", return_value=model) as load:
            with patch.object(qwen_voice_clone, "_write_wav") as write:
                metadata = qwen_voice_clone.run(
                    reference_audio=Path("reference.wav"),
                    reference_text="正確な参照音声の文字起こし",
                    text="生成する文章",
                    output=output,
                )

        load.assert_called_once_with(
            qwen_voice_clone.DEFAULT_MODEL,
            "cuda:0",
            "sdpa",
        )
        model.generate_voice_clone.assert_called_once_with(
            text="生成する文章",
            language="Japanese",
            ref_audio="reference.wav",
            ref_text="正確な参照音声の文字起こし",
            x_vector_only_mode=False,
        )
        write.assert_called_once_with(output, audio, 24000)
        self.assertEqual(metadata["model"], qwen_voice_clone.DEFAULT_MODEL)
        self.assertEqual(metadata["sample_rate"], 24000)
        self.assertEqual(metadata["audio_seconds"], 1.0)
        self.assertIn("RTF", metadata)

    def test_main_emits_json_without_loading_qwen(self):
        with TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "reference.wav"
            reference.write_bytes(b"RIFF")
            output = Path(tmpdir) / "output.wav"
            metadata = {"model": qwen_voice_clone.DEFAULT_MODEL, "output": str(output)}
            stdout = io.StringIO()

            with patch.object(qwen_voice_clone, "run", return_value=metadata) as run:
                exit_code = qwen_voice_clone.main(
                    [
                        "--reference-audio",
                        str(reference),
                        "--reference-text",
                        "参照",
                        "--text",
                        "生成",
                        "--output",
                        str(output),
                    ],
                    stdout=stdout,
                )

        self.assertEqual(exit_code, 0)
        run.assert_called_once()
        self.assertIn('"model"', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
