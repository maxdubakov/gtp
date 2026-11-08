"""Run provenance and W&B tracking for Hydra training scripts.

Every run directory gets `provenance.json` (git state, uv.lock digest, device, host),
`git.diff` when tracked files are dirty, and `train.log`. The same data goes to W&B,
grouped by the config `name` (set by each experiment file).
"""

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.log import set_log_file

OmegaConf.register_new_resolver('repo_root', lambda: str(REPO_ROOT), replace=True)


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def git_state():
    """(HEAD sha, diff of tracked files against HEAD); from VERSION / GIT_DIFF in a packed copy without .git."""
    if not (REPO_ROOT / '.git').exists():
        diff = REPO_ROOT / 'GIT_DIFF'
        return (REPO_ROOT / 'VERSION').read_text().strip(), diff.read_text() if diff.exists() else ''
    return _git('rev-parse', 'HEAD').strip(), _git('diff', 'HEAD')


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


@dataclass
class Run:
    dir: Path
    experiment: str
    name: str

    def log(self, metrics, step=None):
        wandb.log(metrics, step=step)

    def finish(self, summary):
        wandb.summary.update(summary)
        wandb.finish()
        set_log_file(None)


def start_run(cfg, job_type, device, extra=None):
    hc = HydraConfig.get()
    run_dir = Path(hc.runtime.output_dir)
    experiment = cfg.name
    name = f'{experiment}/{hc.job.override_dirname or "defaults"}'

    sha, diff = git_state()
    if diff:
        (run_dir / 'git.diff').write_text(diff)
    provenance = {
        'experiment': experiment,
        'git_sha': sha,
        'git_dirty': bool(diff),
        'uv_lock': file_digest(REPO_ROOT / 'uv.lock'),
        'device': device,
        'torch': torch.__version__,
        'host': platform.node(),
        'started': datetime.now().isoformat(timespec='seconds'),
        **(extra or {}),
    }
    (run_dir / 'provenance.json').write_text(json.dumps(provenance, indent=1))
    set_log_file(run_dir / 'train.log')

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        group=experiment,
        job_type=job_type,
        name=name,
        dir=run_dir,
        config={**OmegaConf.to_container(cfg, resolve=True), 'provenance': provenance},
    )
    return Run(dir=run_dir, experiment=experiment, name=name)
