from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_tokkio_workspace_lifecycle", ROOT / "infra" / "tokkio" / "prepare_tokkio_workspace.py")


class TokkioIrodoriLifecycleTests(unittest.TestCase):
    def test_resolve_irodori_tts_settings_defaults_to_app_host(self) -> None:
        settings = prepare.resolve_irodori_tts_settings(
            {
                "TOKKIO_APP_HOST_IPV4_ADDR": "10.0.0.42",
            }
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.base_url, "http://10.0.0.42:8021")
        self.assertEqual(settings.health_url, "http://127.0.0.1:8021/healthz")
        self.assertEqual(settings.service, "ace-irodori-tts.service")

    def test_manage_script_mentions_irodori_start_stop_status_hooks(self) -> None:
        script = (ROOT / "infra" / "tokkio" / "manage_tokkio.sh").read_text(encoding="utf-8")

        self.assertIn("start_irodori_tts_service", script)
        self.assertIn("stop_irodori_tts_service", script)
        self.assertIn("show_irodori_tts_status", script)
        self.assertIn("TOKKIO_IRODORI_TTS_ENABLED", script)

    def test_manage_script_syncs_controller_to_code_and_app_paths(self) -> None:
        script = (ROOT / "infra" / "tokkio" / "manage_tokkio.sh").read_text(encoding="utf-8")

        self.assertIn("sync-controller", script)
        self.assertIn("sync_controller_runtime_files", script)
        self.assertIn("TOKKIO_CONTROLLER_SOURCE_DIR", script)
        self.assertIn("/code/src", script)
        self.assertIn("/app/src", script)
        self.assertIn("Publishing the verified controller bundle", script)
        self.assertIn("tokkio_irodori_tts.py", script)

    def test_manage_script_does_not_control_local_llm_lifecycle(self) -> None:
        script = (ROOT / "infra" / "tokkio" / "manage_tokkio.sh").read_text(encoding="utf-8")

        self.assertNotIn("start_local_llm_service", script)
        self.assertNotIn("stop_local_llm_service", script)
        self.assertNotIn("show_local_llm_status", script)
        self.assertNotIn("TOKKIO_LLM_SERVICE_ENABLED", script)

    def test_manage_script_protects_replica_snapshot_recovery(self) -> None:
        script = (ROOT / "infra" / "tokkio" / "manage_tokkio.sh").read_text(encoding="utf-8")

        self.assertIn("workload_snapshot_has_positive_replicas", script)
        self.assertIn("Current workload snapshot contains no positive replicas", script)
        self.assertIn("Saved workload snapshot contains no positive replicas", script)
        self.assertIn("rebuild_app_workload_replicas_from_helm_manifest", script)
        self.assertIn("TOKKIO_HELM_RELEASE", script)
        self.assertIn("helm get manifest", script)

    def test_llm_compose_file_uses_configurable_container_name(self) -> None:
        compose_text = (ROOT / "infra" / "llm" / "docker-compose.trtllm.yml").read_text(encoding="utf-8")

        self.assertIn("container_name: ${TRTLLM_CONTAINER_NAME:-ace-trtllm-osaka-swallow}", compose_text)


if __name__ == "__main__":
    unittest.main()
