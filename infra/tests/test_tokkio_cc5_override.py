from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
OVERRIDE_PATH = ROOT / "infra" / "tokkio" / "overrides" / "cc5-unreal-renderer.values.yaml"
RENDERER_KEY = "ia-unreal-renderer-microservice"
EXPECTED_ASR_RMIR = "nvidia/riva/rmir_asr_conformer_unified_ja_jp_str:2.19.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokkioCc5OverrideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not OVERRIDE_PATH.is_file():
            raise AssertionError(f"missing CC5 Helm override: {OVERRIDE_PATH}")
        cls.values = yaml.safe_load(OVERRIDE_PATH.read_text(encoding="utf-8"))
        cls.renderer = cls.values["tokkio-app"][RENDERER_KEY]
        deployment = cls.renderer["applicationSpecs"]["deployment"]
        cls.deployment = deployment
        cls.init = deployment["initContainers"][0]
        cls.script = cls.init["args"][0]

    def test_override_pins_the_japanese_riva_asr_model(self) -> None:
        models = self.values["riva-api"]["modelRepoGenerator"]["ngcModelConfigs"]["triton0"]["models"]
        self.assertEqual(models, [EXPECTED_ASR_RMIR])

    def test_override_uses_local_copy_and_existing_expanded_pvc(self) -> None:
        serialized = yaml.safe_dump(self.renderer)
        self.assertNotIn("ngc-resource-downloader", serialized)
        self.assertEqual(self.init["name"], "stage-cc5-project")
        self.assertEqual(self.init["securityContext"]["runAsUser"], 0)

        mounts = {mount["name"]: mount for mount in self.init["volumeMounts"]}
        self.assertEqual(mounts["asset-volume"]["mountPath"], "/home/unreal-renderer")
        self.assertTrue(mounts["cc5-package-source"]["readOnly"])

        volumes = {volume["name"]: volume for volume in self.deployment["volumes"]}
        self.assertEqual(
            volumes["asset-volume"]["persistentVolumeClaim"]["claimName"],
            "ia-unreal-renderer-microservice-assets",
        )
        self.assertEqual(
            volumes["cc5-package-source"]["hostPath"],
            {
                "path": "/home2/ko66/ace-sandbox/tokkio/unreal-resources/cc5_tokkio/2026-07-15",
                "type": "Directory",
            },
        )
        self.assertEqual(
            self.renderer["storageClaims"]["assets"]["spec"]["resources"]["requests"]["storage"],
            "8Gi",
        )

    def test_override_pins_the_2026_07_15_package_integrity_values(self) -> None:
        env = {item["name"]: item.get("value") for item in self.init["env"]}
        self.assertEqual(env["CC5_VERSION"], "2026-07-15")
        self.assertIn(
            'EXECUTABLE_REL="cc5_tokkio/Binaries/Linux/cc5_tokkio"',
            self.script,
        )
        self.assertEqual(
            env["EXPECTED_LAUNCHER_SHA256"],
            "031bc7c70cd6287bd1cc4d83ab78c830a91ae6b93fc4b161d2041723a036b1b9",
        )
        self.assertEqual(
            env["EXPECTED_EXECUTABLE_SHA256"],
            "b94406c48b585cb5701b7d117ef3667eeaa241db8bf7b7297dd173f65d418f77",
        )
        self.assertEqual(
            env["EXPECTED_UCAS_SHA256"],
            "a34591ba2babf989dec7d928d989b7a08f117397dd3038c731997235e8ad21ba",
        )

    def test_override_aligns_renderer_and_signalling_with_ue_55(self) -> None:
        self.assertEqual(self.renderer["unrealEngine"]["version"], "5.5")
        self.assertEqual(self.renderer["unrealEngine"]["signallingServerVersion"], "5.5")

        containers = self.deployment["containers"]
        self.assertEqual(containers["signalling"]["image"]["tag"], "5.5")
        for name in ("ms", "signalling"):
            env = {item["name"]: item.get("value") for item in containers[name]["env"]}
            self.assertEqual(env["UE_VERSION"], "5.5")
            self.assertEqual(env["UNREAL_ENGINE_VERSION"], "5.5")
            self.assertEqual(env["NVIDIA_VISIBLE_DEVICES"], "1")

        signalling = containers["signalling"]
        self.assertEqual(signalling["command"], ["/bin/bash", "-c"])
        command_text = signalling["args"][0]
        self.assertIn("--streamer_port=", command_text)
        self.assertIn("--player_port=", command_text)
        self.assertIn("--peer_options=", command_text)
        self.assertNotIn("--HttpPort", command_text)
        self.assertNotIn("--peerConnectionOptions", command_text)

    def test_override_renders_cc5_at_1080p(self) -> None:
        ms_env = {
            item["name"]: item.get("value")
            for item in self.deployment["containers"]["ms"]["env"]
        }
        self.assertEqual(ms_env["IAUEMS_WINDOW_WIDTH"], "1920")
        self.assertEqual(ms_env["IAUEMS_WINDOW_HEIGHT"], "1080")

    def test_override_uses_a_conservative_cc5_face_profile(self) -> None:
        self.assertIn("audio2face-3d", self.values)
        a2f = self.values["audio2face-3d"]["a2f"]["configs"][
            "stylization_config.yaml"
        ]["a2f"]

        # Keep the supported A2F model/output topology, but do not apply the
        # James-specific tongue and face-strength tuning to the CC5 ARKit rig.
        self.assertEqual(a2f["inference_model_id"], "james_v2.3")
        self.assertEqual(a2f["blendshape_id"], "james_topo2_v2.3")
        self.assertFalse(a2f["enable_tongue_blendshapes"])

        self.assertEqual(
            a2f["face_params"],
            {
                "lip_close_offset": 0.0,
                "lip_open_offset": 0.0,
                "lower_face_smoothing": 0.015,
                "lower_face_strength": 0.9,
                "skin_strength": 1.0,
            },
        )

        blendshape_params = a2f["blendshape_params"]
        self.assertTrue(blendshape_params["enable_clamping_bs_weight"])
        self.assertEqual(
            blendshape_params["weight_multipliers"],
            {
                "JawForward": 1.0,
                "JawLeft": 1.0,
                "JawOpen": 1.0,
                "JawRight": 1.0,
                "MouthClose": 1.0,
                "MouthDimpleLeft": 1.0,
                "MouthDimpleRight": 1.0,
                "MouthFrownLeft": 1.0,
                "MouthFrownRight": 1.0,
                "MouthFunnel": 1.0,
                "MouthLeft": 1.0,
                "MouthLowerDownLeft": 1.0,
                "MouthLowerDownRight": 1.0,
                "MouthPressLeft": 1.0,
                "MouthPressRight": 1.0,
                "MouthPucker": 1.0,
                "MouthRight": 1.0,
                "MouthRollLower": 1.0,
                "MouthRollUpper": 1.0,
                "MouthShrugLower": 1.0,
                "MouthShrugUpper": 1.0,
                "MouthSmileLeft": 1.0,
                "MouthSmileRight": 1.0,
                "MouthStretchLeft": 1.0,
                "MouthStretchRight": 1.0,
                "MouthUpperUpLeft": 1.0,
                "MouthUpperUpRight": 1.0,
                "TongueOut": 0.0,
            },
        )

    def test_controller_config_registers_official_user_override(self) -> None:
        config_path = ROOT / "infra" / "tokkio" / "workspace" / "controller" / "ace-app-config.yml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        override_files = config["spec"]["app"]["configs"]["app_settings"]["helm_chart"]["repo"][
            "user_value_override_files"
        ]
        self.assertEqual(override_files, [str(OVERRIDE_PATH)])

    def test_init_script_contains_integrity_capacity_idempotency_and_rollback_guards(self) -> None:
        required_fragments = (
            "PVC_CAPACITY_BYTES",
            "PVC_SAFETY_MARGIN_BYTES",
            "EXPECTED_LAUNCHER_SHA256",
            "EXPECTED_EXECUTABLE_SHA256",
            "EXPECTED_UCAS_SHA256",
            "unrealEngineProject.next",
            "unrealEngineProject.prev",
            ".cc5-package-version",
            "already installed and verified",
            "rollback",
            "sha256sum",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, self.script)

    def _make_fixture(self, root: Path) -> tuple[Path, Path, dict[str, str]]:
        source = root / "source"
        asset_root = root / "assets"
        current = asset_root / "unrealEngineProject"
        (source / "cc5_tokkio" / "Binaries" / "Linux").mkdir(parents=True)
        (source / "cc5_tokkio" / "Content" / "Paks").mkdir(parents=True)
        current.mkdir(parents=True)

        launcher = source / "cc5_tokkio.sh"
        executable = source / "cc5_tokkio" / "Binaries" / "Linux" / "cc5_tokkio"
        ucas = source / "cc5_tokkio" / "Content" / "Paks" / "cc5_tokkio-Linux.ucas"
        launcher.write_text("#!/bin/sh\necho cc5\n", encoding="utf-8")
        executable.write_bytes(b"fake-elf-for-staging-test")
        ucas.write_bytes(b"fake-cooked-payload-for-staging-test")
        launcher.chmod(0o755)
        executable.chmod(0o755)
        (current / "aki.txt").write_text("previous project", encoding="utf-8")

        env = os.environ.copy()
        env.update(
            {
                "SOURCE_DIR": str(source),
                "ASSET_ROOT": str(asset_root),
                "CC5_VERSION": "test-version",
                "PVC_CAPACITY_BYTES": str(1024 * 1024 * 1024),
                "PVC_SAFETY_MARGIN_BYTES": "0",
                "EXPECTED_LAUNCHER_SHA256": sha256(launcher),
                "EXPECTED_EXECUTABLE_SHA256": sha256(executable),
                "EXPECTED_UCAS_SHA256": sha256(ucas),
                "TARGET_UID": str(os.getuid()),
                "TARGET_GID": str(os.getgid()),
            }
        )
        return source, asset_root, env

    def _run_script(self, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", self.script],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_init_script_stages_promotes_and_skips_verified_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, asset_root, env = self._make_fixture(Path(tmp))
            first = self._run_script(env)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)

            current = asset_root / "unrealEngineProject"
            previous = asset_root / "unrealEngineProject.prev"
            self.assertEqual((current / "cc5_tokkio.sh").read_bytes(), (source / "cc5_tokkio.sh").read_bytes())
            self.assertTrue((previous / "aki.txt").is_file())
            inode = current.stat().st_ino

            second = self._run_script(env)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("already installed and verified", second.stdout)
            self.assertEqual(current.stat().st_ino, inode)

    def test_source_hash_failure_preserves_current_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, asset_root, env = self._make_fixture(Path(tmp))
            current = asset_root / "unrealEngineProject"
            (source / "cc5_tokkio" / "Content" / "Paks" / "cc5_tokkio-Linux.ucas").write_bytes(b"tampered")

            completed = self._run_script(env)
            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue((current / "aki.txt").is_file())
            self.assertFalse((asset_root / "unrealEngineProject.next").exists())

    def test_capacity_failure_preserves_current_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _source, asset_root, env = self._make_fixture(Path(tmp))
            current = asset_root / "unrealEngineProject"
            env["PVC_CAPACITY_BYTES"] = "1"

            completed = self._run_script(env)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("insufficient logical PVC capacity", completed.stdout + completed.stderr)
            self.assertTrue((current / "aki.txt").is_file())


if __name__ == "__main__":
    unittest.main()
