import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from ace_tts.cli import main


class CliTests(unittest.TestCase):
    def test_download_reference_audio_invokes_downloader_with_expected_defaults(self):
        with TemporaryDirectory() as tmpdir:
            download = Mock(return_value=Path(tmpdir) / "reference.wav")
            stdout = io.StringIO()

            exit_code = main(
                [
                    "download-reference",
                    "https://www.youtube.com/watch?v=example",
                    "--output-dir",
                    tmpdir,
                ],
                download_audio=download,
                stdout=stdout,
            )

        self.assertEqual(exit_code, 0)
        download.assert_called_once_with(
            "https://www.youtube.com/watch?v=example",
            output_dir=Path(tmpdir),
            output_name="reference",
            start=None,
            duration=None,
        )
        self.assertIn("reference.wav", stdout.getvalue())

    def test_download_reference_audio_accepts_start_and_duration(self):
        with TemporaryDirectory() as tmpdir:
            download = Mock(return_value=Path(tmpdir) / "reference.wav")

            exit_code = main(
                [
                    "download-reference",
                    "https://www.youtube.com/watch?v=example",
                    "--output-dir",
                    tmpdir,
                    "--start",
                    "12.5",
                    "--duration",
                    "8",
                ],
                download_audio=download,
            )

        self.assertEqual(exit_code, 0)
        download.assert_called_once_with(
            "https://www.youtube.com/watch?v=example",
            output_dir=Path(tmpdir),
            output_name="reference",
            start=12.5,
            duration=8.0,
        )

    def test_synthesize_requires_existing_reference_audio(self):
        stderr = io.StringIO()

        exit_code = main(
            [
                "synthesize",
                "--reference-audio",
                "missing.wav",
                "--reference-text",
                "こんにちは",
                "--text",
                "テストです",
            ],
            synthesize=Mock(),
            stderr=stderr,
        )

        self.assertEqual(exit_code, 2)
        self.assertIn("reference audio does not exist", stderr.getvalue())

    def test_synthesize_invokes_generator_with_text_inputs(self):
        with TemporaryDirectory() as tmpdir:
            ref_audio = Path(tmpdir) / "reference.wav"
            ref_audio.write_bytes(b"RIFF....WAVE")
            output = Path(tmpdir) / "voice.wav"
            synthesize = Mock(return_value=output)
            stdout = io.StringIO()

            exit_code = main(
                [
                    "synthesize",
                    "--reference-audio",
                    str(ref_audio),
                    "--reference-text",
                    "これは参照音声です",
                    "--text",
                    "生成したい文章です",
                    "--output",
                    str(output),
                ],
                synthesize=synthesize,
                stdout=stdout,
            )

        self.assertEqual(exit_code, 0)
        synthesize.assert_called_once_with(
            reference_audio=ref_audio,
            reference_text="これは参照音声です",
            text="生成したい文章です",
            output_path=output,
            model_name="sbintuitions/sarashina2.2-tts",
        )
        self.assertIn("voice.wav", stdout.getvalue())

    def test_clone_from_youtube_downloads_then_synthesizes(self):
        with TemporaryDirectory() as tmpdir:
            ref_audio = Path(tmpdir) / "reference.wav"
            ref_audio.write_bytes(b"RIFF....WAVE")
            output = Path(tmpdir) / "cloned.wav"
            download = Mock(return_value=ref_audio)
            synthesize = Mock(return_value=output)

            exit_code = main(
                [
                    "clone-from-youtube",
                    "https://www.youtube.com/watch?v=example",
                    "--reference-text",
                    "参照音声の文字起こしです",
                    "--text",
                    "この声で読み上げます",
                    "--work-dir",
                    tmpdir,
                    "--output",
                    str(output),
                ],
                download_audio=download,
                synthesize=synthesize,
            )

        self.assertEqual(exit_code, 0)
        download.assert_called_once_with(
            "https://www.youtube.com/watch?v=example",
            output_dir=Path(tmpdir),
            output_name="reference",
            start=None,
            duration=None,
        )
        synthesize.assert_called_once_with(
            reference_audio=ref_audio,
            reference_text="参照音声の文字起こしです",
            text="この声で読み上げます",
            output_path=output,
            model_name="sbintuitions/sarashina2.2-tts",
        )


if __name__ == "__main__":
    unittest.main()
