"""Tests for so101_tool.training — pure command building, device fallback, Colab export.

These run on the sim-only Python 3.11 venv: no lerobot, no torch required.
"""

import builtins
import json

import pytest

from so101_tool.training import (
    LEROBOT_TRAIN,
    TrainArgs,
    build_train_command,
    emit_colab,
    push_dataset,
    resolve_device,
    train_cli,
    train_local,
)


def _args(**kw):
    return TrainArgs(dataset="user/so101-pick-cube", **kw)


class TestBuildTrainCommand:
    def test_hub_dataset_flags(self):
        cmd = build_train_command(_args(device="cpu"))
        assert cmd[0] == LEROBOT_TRAIN
        assert "--dataset.repo_id=user/so101-pick-cube" in cmd
        assert not any(c.startswith("--dataset.root=") for c in cmd)
        assert "--policy.type=act" in cmd
        assert "--policy.device=cpu" in cmd
        assert "--output_dir=outputs/train" in cmd
        assert "--steps=20000" in cmd
        assert "--batch_size=8" in cmd

    def test_wandb_disabled_and_no_hub_push_by_default(self):
        cmd = build_train_command(_args(device="cpu"))
        assert "--wandb.enable=false" in cmd
        assert "--policy.push_to_hub=false" in cmd

    def test_local_dataset_path_uses_dataset_root(self, tmp_path):
        ds = tmp_path / "my_demos"
        ds.mkdir()
        cmd = build_train_command(TrainArgs(dataset=str(ds), device="cpu"))
        assert f"--dataset.root={ds}" in cmd
        assert "--dataset.repo_id=local/my_demos" in cmd

    def test_policy_and_hyperparams(self):
        cmd = build_train_command(
            _args(policy="smolvla", steps=500, batch_size=2, output_dir="out/x", device="cpu")
        )
        assert "--policy.type=smolvla" in cmd
        assert "--steps=500" in cmd
        assert "--batch_size=2" in cmd
        assert "--output_dir=out/x" in cmd

    def test_extra_passthrough_appended_verbatim(self):
        extra = ["--save_freq=1000", "--wandb.enable=true"]
        cmd = build_train_command(_args(device="cpu", extra=extra))
        assert cmd[-2:] == extra

    def test_push_to_hub_requires_hub_repo(self):
        with pytest.raises(ValueError, match="hub_repo"):
            build_train_command(_args(device="cpu", push_to_hub=True))

    def test_push_to_hub_flags(self):
        cmd = build_train_command(
            _args(device="cpu", push_to_hub=True, hub_repo="user/so101-act")
        )
        assert "--policy.push_to_hub=true" in cmd
        assert "--policy.repo_id=user/so101-act" in cmd

    def test_explicit_dataset_root_wins(self, tmp_path):
        ds = tmp_path / "demos"
        ds.mkdir()
        cmd = build_train_command(
            _args(device="cpu", dataset_root=str(ds))  # dataset stays a hub repo_id
        )
        assert f"--dataset.root={ds}" in cmd
        assert "--dataset.repo_id=user/so101-pick-cube" in cmd

    def test_dataset_root_with_local_dataset_uses_synthetic_repo_id(self, tmp_path):
        ds = tmp_path / "demos"
        ds.mkdir()
        cmd = build_train_command(
            TrainArgs(dataset=str(ds), dataset_root=str(ds), device="cpu")
        )
        assert f"--dataset.root={ds}" in cmd
        assert "--dataset.repo_id=local/demos" in cmd

    def test_resume_adds_config_path_under_output_dir(self):
        cmd = build_train_command(_args(device="cpu", resume=True, output_dir="out/run1"))
        assert "--resume=true" in cmd
        assert (
            "--config_path=out/run1/checkpoints/last/pretrained_model/train_config.json" in cmd
        )


class TestResolveDevice:
    def test_explicit_device_passes_through(self):
        assert resolve_device("cuda:1") == "cuda:1"

    def test_auto_falls_back_to_cpu_without_torch(self, monkeypatch):
        real_import = builtins.__import__

        def no_torch(name, *a, **kw):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("No module named 'torch'")
            return real_import(name, *a, **kw)

        monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)
        monkeypatch.setattr(builtins, "__import__", no_torch)
        assert resolve_device("auto") == "cpu"

    def test_auto_resolves_in_command(self, monkeypatch):
        import so101_tool.training as training

        monkeypatch.setattr(training, "resolve_device", lambda d="auto": "cpu")
        cmd = training.build_train_command(_args())  # device="auto"
        assert "--policy.device=cpu" in cmd
        assert not any("device=auto" in c for c in cmd)


class TestTrainLocal:
    def test_friendly_error_when_lerobot_train_missing(self, monkeypatch):
        import so101_tool.training as training

        monkeypatch.setattr(training.shutil, "which", lambda _: None)
        with pytest.raises(RuntimeError) as exc:
            train_local(_args(device="cpu"))
        msg = str(exc.value)
        assert 'pip install "so101-tool[policy]"' in msg
        assert "3.12" in msg

    def test_runs_subprocess_and_returns_exit_code(self, monkeypatch):
        import so101_tool.training as training

        seen = {}

        class FakeCompleted:
            returncode = 7

        def fake_run(argv, **kw):
            seen["argv"] = argv
            return FakeCompleted()

        monkeypatch.setattr(training.shutil, "which", lambda _: "/fake/bin/lerobot-train")
        monkeypatch.setattr(training.subprocess, "run", fake_run)
        assert train_local(_args(device="cpu")) == 7
        assert seen["argv"][0] == "/fake/bin/lerobot-train"
        assert "--dataset.repo_id=user/so101-pick-cube" in seen["argv"]


class TestPushDataset:
    def test_rejects_non_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            push_dataset(str(tmp_path / "nope"), "user/ds")


class TestEmitColab:
    def test_writes_valid_nbformat4_json_with_train_command(self, tmp_path):
        out = tmp_path / "train.ipynb"
        path = emit_colab(_args(device="cpu"), str(out))
        assert path == out
        nb = json.loads(out.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert isinstance(nb["cells"], list) and len(nb["cells"]) >= 5
        for cell in nb["cells"]:
            assert cell["cell_type"] in ("markdown", "code")
            assert "source" in cell and "metadata" in cell
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                assert cell["execution_count"] is None
        sources = "\n".join(
            c["source"] if isinstance(c["source"], str) else "".join(c["source"])
            for c in nb["cells"]
        )
        assert "!lerobot-train" in sources
        assert "--dataset.repo_id=user/so101-pick-cube" in sources
        assert "--policy.device=cuda" in sources  # Colab GPU, regardless of local device
        assert 'pip install "lerobot>=0.5.1"' in sources
        assert "notebook_login" in sources
        assert "upload_folder" in sources  # checkpoint push cell

    def test_dataset_hub_repo_overrides_local_path(self, tmp_path):
        ds = tmp_path / "demos"
        ds.mkdir()
        out = tmp_path / "nb.ipynb"
        emit_colab(TrainArgs(dataset=str(ds), device="cpu"), str(out), "user/uploaded-demos")
        text = out.read_text(encoding="utf-8")
        assert "user/uploaded-demos" in text
        assert str(ds) not in text

    def test_rejects_local_dataset_without_hub_repo(self, tmp_path):
        ds = tmp_path / "demos"
        ds.mkdir()
        with pytest.raises(ValueError, match="[Pp]ush"):
            emit_colab(TrainArgs(dataset=str(ds)), str(tmp_path / "nb.ipynb"))

    def test_notebook_never_references_local_dataset_root(self, tmp_path):
        ds = tmp_path / "demos"
        ds.mkdir()
        out = tmp_path / "nb.ipynb"
        emit_colab(
            TrainArgs(dataset="user/hub-ds", dataset_root=str(ds), device="cpu"), str(out)
        )
        text = out.read_text(encoding="utf-8")
        assert "--dataset.root" not in text
        assert "--dataset.repo_id=user/hub-ds" in text


class TestTrainCli:
    def test_emit_colab_writes_notebook_and_returns(self, tmp_path, monkeypatch, capsys):
        import so101_tool.training as training

        # train_local must NOT be called in emit-colab mode
        monkeypatch.setattr(
            training, "train_local", lambda a: (_ for _ in ()).throw(AssertionError)
        )
        out = tmp_path / "train.ipynb"
        train_cli(_args(device="cpu", emit_colab=str(out)))
        assert out.exists()
        assert "colab.research.google.com" in capsys.readouterr().out

    def test_push_dataset_requires_hub_repo(self, tmp_path):
        ds = tmp_path / "demos"
        ds.mkdir()
        with pytest.raises(SystemExit, match="dataset-hub-repo"):
            train_cli(TrainArgs(dataset=str(ds), push_dataset=True, device="cpu"))

    def test_push_then_emit_colab(self, tmp_path, monkeypatch):
        import so101_tool.training as training

        ds = tmp_path / "demos"
        ds.mkdir()
        pushed = {}
        monkeypatch.setattr(
            training, "push_dataset", lambda root, repo: pushed.update(root=root, repo=repo)
        )
        out = tmp_path / "train.ipynb"
        train_cli(
            TrainArgs(
                dataset=str(ds),
                device="cpu",
                push_dataset=True,
                dataset_hub_repo="user/uploaded",
                emit_colab=str(out),
            )
        )
        assert pushed == {"root": str(ds), "repo": "user/uploaded"}
        assert "user/uploaded" in out.read_text(encoding="utf-8")

    def test_train_local_nonzero_exit_becomes_systemexit(self, monkeypatch):
        import so101_tool.training as training

        monkeypatch.setattr(training, "train_local", lambda a: 3)
        with pytest.raises(SystemExit):
            train_cli(_args(device="cpu"))
        monkeypatch.setattr(training, "train_local", lambda a: 0)
        train_cli(_args(device="cpu"))  # rc == 0 -> no exception

    def test_missing_lerobot_is_clean_systemexit(self, monkeypatch):
        import so101_tool.training as training

        monkeypatch.setattr(training.shutil, "which", lambda _: None)
        with pytest.raises(SystemExit, match=r"so101-tool\[policy\]"):
            train_cli(_args(device="cpu"))
