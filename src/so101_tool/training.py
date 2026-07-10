"""Policy training through so101-tool: local ``lerobot-train`` runs and cloud notebooks.

This module is a thin, dependency-free front-end over lerobot's training entry point
(``lerobot-train``, draccus-style CLI, lerobot >= 0.5.1):

- :func:`build_train_command` maps :class:`TrainArgs` onto the real ``lerobot-train``
  flags (``--dataset.repo_id`` / ``--dataset.root``, ``--policy.type``, ``--steps``,
  ``--batch_size``, ``--output_dir``, ...). Pure and testable without lerobot installed.
- :func:`train_local` runs that command as a subprocess (requires the ``[policy]``
  extra, Python >= 3.12).
- :func:`push_dataset` uploads a recorded local LeRobotDataset directory to the
  Hugging Face Hub so it can be used from any machine (requires ``HF_TOKEN``).
- :func:`emit_colab` writes a ready-to-run Colab notebook that trains the same
  configuration on a free cloud GPU.

Nothing here imports torch or lerobot at module import time, so the module works on
the sim-only Python 3.10/3.11 install too.
"""

from __future__ import annotations

import dataclasses
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import field
from pathlib import Path

LEROBOT_TRAIN = "lerobot-train"

#: Policy types we document and test against (lerobot accepts more, e.g. "pi0";
#: unknown types are passed through and validated by lerobot itself).
KNOWN_POLICIES = ("act", "smolvla", "diffusion")

_MISSING_LEROBOT_MSG = (
    "'lerobot-train' was not found on PATH. Training needs the lerobot training stack, "
    "which requires Python >= 3.12. Install it with:\n"
    '    pip install "so101-tool[policy]"\n'
    "(or 'pip install \"lerobot>=0.5.1\"') and re-run from that environment."
)


@dataclasses.dataclass
class TrainArgs:
    """Everything needed to launch one training run (locally or in the cloud)."""

    dataset: str
    """HF hub repo_id (e.g. "user/so101-pick-cube") or local dataset directory."""
    policy: str = "act"
    """Policy type to train: act | smolvla | diffusion."""
    output_dir: str = "outputs/train"
    """Where checkpoints are written (<output_dir>/checkpoints/...)."""
    steps: int = 20000
    """Number of training steps."""
    batch_size: int = 8
    """Training batch size."""
    device: str = "auto"
    """Compute device: cuda, mps, cpu, or auto (cuda if available, else mps, else cpu)."""
    resume: bool = False
    """Resume training from <output_dir>/checkpoints/last."""
    push_to_hub: bool = False
    """Push the trained policy to the HF hub (requires --hub-repo)."""
    hub_repo: str | None = None
    """Target hub repo_id for the pushed policy (e.g. "user/so101-act-pick-cube")."""
    dataset_root: str | None = None
    """Local dataset directory passed to lerobot-train as --dataset.root (e.g. the
    output of scripted-demos); --dataset is then used as the repo_id."""
    emit_colab: str | None = None
    """Instead of training, write a ready-to-run Colab notebook to this path."""
    dataset_hub_repo: str | None = None
    """Hub dataset repo_id used inside the emitted notebook and as --push-dataset target."""
    push_dataset: bool = False
    """Upload the local dataset (--dataset-root or --dataset) to --dataset-hub-repo
    before training / notebook emission."""
    extra: list[str] = field(default_factory=list)
    """Raw passthrough arguments appended verbatim to lerobot-train."""


def resolve_device(device: str = "auto") -> str:
    """Resolve "auto" to cuda/mps/cpu. Torch is imported lazily; no torch means cpu."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _looks_local(dataset: str) -> bool:
    return Path(dataset).expanduser().exists() or dataset.startswith((".", "/", "~"))


def _dataset_flags(dataset: str, root: str | None = None) -> list[str]:
    """Map a dataset spec onto lerobot's --dataset.* flags.

    A hub repo_id ("user/name") is passed as ``--dataset.repo_id``. A local dataset
    directory (recorded with ``so101-tool run --record`` / ``scripted-demos``) is passed
    as ``--dataset.root=<dir>`` with a synthetic repo_id, which is how lerobot 0.5+/0.6
    addresses concrete local dataset trees (repo_id is only used for lookup/naming when
    root is not given). An explicit ``root`` wins over path autodetection.
    """
    if root is not None:
        root_path = Path(root).expanduser()
        repo_id = dataset if not _looks_local(dataset) else f"local/{root_path.name}"
        return [f"--dataset.repo_id={repo_id}", f"--dataset.root={root_path}"]
    path = Path(dataset).expanduser()
    if _looks_local(dataset):
        return [f"--dataset.repo_id=local/{path.name}", f"--dataset.root={path}"]
    return [f"--dataset.repo_id={dataset}"]


def build_train_command(args: TrainArgs) -> list[str]:
    """Build the exact ``lerobot-train`` argv for ``args`` (pure; safe without lerobot).

    Flag names verified against lerobot 0.6.0's ``TrainPipelineConfig`` (draccus CLI):
    ``--dataset.repo_id`` / ``--dataset.root``, ``--policy.type``, ``--policy.device``,
    ``--policy.push_to_hub`` / ``--policy.repo_id``, ``--output_dir``, ``--steps``,
    ``--batch_size``, ``--wandb.enable``, ``--resume`` + ``--config_path``.
    """
    if args.push_to_hub and not args.hub_repo:
        raise ValueError(
            "push_to_hub=True requires hub_repo (e.g. 'your-hf-user/so101-act-pick-cube')"
        )
    device = resolve_device(args.device)
    cmd = [
        LEROBOT_TRAIN,
        *_dataset_flags(args.dataset, args.dataset_root),
        f"--policy.type={args.policy}",
        f"--policy.device={device}",
        f"--output_dir={args.output_dir}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        # lerobot defaults policy.push_to_hub to true and then errors without a repo_id,
        # so always state it explicitly. wandb is opt-in via `extra`.
        "--wandb.enable=false",
    ]
    if args.push_to_hub:
        cmd += ["--policy.push_to_hub=true", f"--policy.repo_id={args.hub_repo}"]
    else:
        cmd += ["--policy.push_to_hub=false"]
    if args.resume:
        config_path = (
            Path(args.output_dir) / "checkpoints" / "last" / "pretrained_model"
            / "train_config.json"
        )
        cmd += ["--resume=true", f"--config_path={config_path}"]
    cmd += list(args.extra)
    return cmd


def train_local(args: TrainArgs) -> int:
    """Run ``lerobot-train`` as a subprocess, streaming its output; return the exit code.

    Raises RuntimeError with install instructions if ``lerobot-train`` is not on PATH.
    """
    cmd = build_train_command(args)
    exe = shutil.which(cmd[0])
    if exe is None:
        raise RuntimeError(_MISSING_LEROBOT_MSG)
    print("+ " + shlex.join(cmd), flush=True)
    # Inherit stdout/stderr so training logs stream straight to the terminal.
    return subprocess.run([exe, *cmd[1:]]).returncode


def push_dataset(root_or_repo: str, hub_repo: str) -> None:
    """Upload a local LeRobotDataset directory to the Hugging Face Hub as ``hub_repo``.

    ``root_or_repo`` must be a local dataset directory (the ``--record`` output, i.e. a
    tree containing ``meta/``, ``data/`` and ``videos/``). Authentication uses the
    ``HF_TOKEN`` environment variable (or a prior ``hf auth login``); set it to a token
    with write access before calling.
    """
    root = Path(root_or_repo).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"'{root_or_repo}' is not a local dataset directory. push_dataset uploads a "
            "recorded dataset tree (containing meta/, data/, videos/) to the hub."
        )
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "huggingface_hub is required to push datasets: pip install huggingface_hub "
            "(and set HF_TOKEN to a write token)"
        ) from e
    api = HfApi()
    api.create_repo(hub_repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(root), repo_id=hub_repo, repo_type="dataset")
    print(f"Pushed {root} -> https://huggingface.co/datasets/{hub_repo}")


def _md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text,
    }


def emit_colab(args: TrainArgs, out_path: str, dataset_hub_repo: str | None = None) -> Path:
    """Write a ready-to-run Colab notebook that trains ``args`` on a free cloud GPU.

    ``dataset_hub_repo`` overrides ``args.dataset`` for the notebook (Colab cannot see
    your local disk, so a local dataset must first be uploaded with :func:`push_dataset`).
    Returns the path of the written .ipynb.
    """
    dataset = dataset_hub_repo or args.dataset
    if Path(dataset).expanduser().exists() or dataset.startswith((".", "/", "~")):
        raise ValueError(
            f"'{dataset}' looks like a local path, which Colab cannot access. Push it first "
            "(push_dataset / `so101-tool train --push-dataset`) and pass the hub repo id."
        )
    # Colab GPU runtimes are CUDA; resolve the device now so the notebook is
    # deterministic. Local-only fields (dataset_root) are dropped: Colab pulls the
    # dataset from the hub.
    nb_args = dataclasses.replace(args, dataset=dataset, device="cuda", dataset_root=None)
    train_cmd = shlex.join(build_train_command(nb_args))

    checkpoint_dir = f"{nb_args.output_dir}/checkpoints/last/pretrained_model"
    policy_repo = nb_args.hub_repo or "YOUR_HF_USER/so101-" + nb_args.policy
    push_source = (
        f'''from huggingface_hub import HfApi

api = HfApi()
repo_id = "{policy_repo}"  # <-- where to publish the trained policy
api.create_repo(repo_id, exist_ok=True)
api.upload_folder(folder_path="{checkpoint_dir}", repo_id=repo_id)
print(f"pushed -> https://huggingface.co/{{repo_id}}")'''
    )

    cells = [
        _md_cell(
            f"# so101-tool cloud training ({nb_args.policy})\n\n"
            f"Generated by `so101-tool` on your machine. This notebook trains a "
            f"**{nb_args.policy}** policy on the dataset "
            f"[`{dataset}`](https://huggingface.co/datasets/{dataset}) "
            f"using `lerobot-train`, then pushes the checkpoint to the Hugging Face Hub.\n\n"
            "**Enable the GPU first**: *Runtime → Change runtime type → T4 GPU* "
            "(free tier). Training on the Colab CPU is 10-50x slower.\n\n"
            "Run the cells top to bottom."
        ),
        _code_cell('!pip install "lerobot>=0.5.1"'),
        _md_cell(
            "Log in to Hugging Face (needed to download private datasets and to push "
            "the trained policy). Use a token with *write* access."
        ),
        _code_cell("from huggingface_hub import notebook_login\n\nnotebook_login()"),
        _md_cell(
            f"Train ({nb_args.steps} steps, batch size {nb_args.batch_size}, device "
            "`cuda`). On a T4 this takes roughly 1-4 h for ACT at 20k steps; Colab free "
            "tier sessions can disconnect, so consider lowering `--steps` or resuming."
        ),
        _code_cell(f"!{train_cmd}"),
        _md_cell(
            "Upload the trained checkpoint to the Hub so you can run it back home with\n"
            "`so101-tool run --backend real --policy-path <repo_id>`."
            + (
                ""
                if nb_args.hub_repo
                else " Edit `repo_id` below before running."
            )
        ),
        _code_cell(push_source),
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path} — upload it to https://colab.research.google.com/ and run all cells.")
    return path


def train_cli(args: TrainArgs) -> None:
    """CLI entry point behind ``so101-tool train``: dispatch on the TrainArgs mode.

    Order: (1) push the local dataset to the hub if requested, (2) emit a Colab
    notebook and stop if requested, (3) otherwise train locally and exit non-zero on
    failure. All expected errors surface as clean SystemExit messages (no tracebacks).
    """
    if args.push_dataset:
        if not args.dataset_hub_repo:
            raise SystemExit(
                "--push-dataset needs --dataset-hub-repo (the hub dataset repo_id to "
                "upload to, e.g. 'your-hf-user/so101-pick-cube')."
            )
        local = args.dataset_root or args.dataset
        print(f"Pushing local dataset {local} to hub repo {args.dataset_hub_repo} ...")
        try:
            push_dataset(local, args.dataset_hub_repo)
        except (FileNotFoundError, RuntimeError) as e:
            raise SystemExit(str(e)) from e

    if args.emit_colab:
        try:
            path = emit_colab(args, args.emit_colab, args.dataset_hub_repo)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        print(
            f"\nColab notebook written to {path}. Next steps:\n"
            "  1. Open https://colab.research.google.com/ and upload the notebook\n"
            "     (File -> Upload notebook).\n"
            "  2. Runtime -> Change runtime type -> T4 GPU.\n"
            "  3. Runtime -> Run all, and paste an HF write token when prompted."
        )
        return

    try:
        rc = train_local(args)
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":  # pragma: no cover - convenience for manual testing
    print(shlex.join(build_train_command(TrainArgs(dataset=sys.argv[1]))))
