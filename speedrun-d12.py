import os
import subprocess
import time
from pathlib import Path
import modal

# ---- Configuration ----
APP_NAME = "nanochat-speedrun-h100"
VOLUME_NAME = "nanochat-persistent-storage"
GPU_CONFIG = "H100:8"  # Adjust based on your Modal setup (e.g., "A100:4" or "H100:8")

# Define the app and persistent volume
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Volume layout
VOL_PATH = Path("/vol")
RUNS_DIR = VOL_PATH / "runs"
DATA_DIR = VOL_PATH / "data"

# Image: Standard Debian with build tools + uv (used by nanochat)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "curl", "wget", "build-essential", "libssl-dev", "pkg-config"
    )
    .pip_install("uv", "torch") # Basic python tools
)

# ---- Helper Functions ----

def _run(cmd: str, cwd: Path = None, env: dict = None):
    """Executes a shell command and streams output."""
    print(f"\n[EXEC] {cmd}\n", flush=True)
    subprocess.run(
        ["bash", "-lc", cmd],
        cwd=cwd,
        env=env,
        check=True
    )

def _setup_workspace(model: str, repo_url: str, repo_ref: str) -> Path:
    """
    Clones repo and sets up symlinks for persistence.
    Returns the path to the repository root.
    """
    workdir = Path("/root/nanochat_work")
    repo_dir = workdir / "nanochat"
    
    # 1. Prepare Volume Directories
    _run(f"mkdir -p '{RUNS_DIR}' '{DATA_DIR}'")
    
    # 2. Clone Repo (if not exists in ephemeral container)
    if not repo_dir.exists():
        _run(f"mkdir -p '{workdir}'")
        _run(f"git clone --depth 1 '{repo_url}' '{repo_dir}'")
    
    # 3. Checkout specific ref
    _run(f"git fetch --all --tags && git checkout '{repo_ref}'", cwd=repo_dir)

    # 4. SYMLINK persistence
    # We delete the repo's empty folders and link to our Volume folders
    # This ensures 'data/' and 'logs/' (or 'runs/') are stored on the Volume.
    
    # Link Data
    _run(f"rm -rf data && ln -s {DATA_DIR} data", cwd=repo_dir)
    
    # Link Logs/Outputs (Nanochat typically uses 'logs' or 'out')
    # We create a specific folder for this model to keep things clean
    model_log_dir = RUNS_DIR / model
    _run(f"mkdir -p '{model_log_dir}'")
    _run(f"rm -rf logs && ln -s {RUNS_DIR} logs", cwd=repo_dir)
    
    return repo_dir

# ---- Main Functions ----

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=24 * 60 * 60,  # 24 hours
    volumes={str(VOL_PATH): vol},
    _allow_background_volume_commits=True
)
def run_speedrun(
    model: str = "d12",
    repo_ref: str = "master",
    repo_url: str = "https://github.com/karpathy/nanochat.git",
    force_restart: bool = False
):
    """
    Main training entrypoint. 
    Auto-detects checkpoints to resume, otherwise runs speedrun.sh.
    """
    repo_dir = _setup_workspace(model, repo_url, repo_ref)
    
    # Check for existing checkpoint in the persistent volume
    # Adjust 'ckpt.pt' based on exact file naming in nanochat (often 'ckpt.pt' or 'last.pt')
    ckpt_path = RUNS_DIR / model / "ckpt.pt"
    
    cmd = ""
    if ckpt_path.exists() and not force_restart:
        print(f"\n>>> CHECKPOINT FOUND at {ckpt_path}. RESUMING... <<<\n")
        # Resume command: bypasses speedrun.sh to run train.py directly
        # You may need to add --batch_size adjustment here for 8x GPUs if not in config
        cmd = f"./uv run train.py config/{model}.py --init_from='resume'"
    else:
        print(f"\n>>> STARTING FRESH SPEEDRUN ({model}) <<<\n")
        if force_restart:
            print("Force restart requested: Cleaning old logs...")
            _run(f"rm -rf {RUNS_DIR}/{model}/*")
            
        # Standard entry point
        cmd = f"./speedrun.sh {model}"

    # Execute
    try:
        # We assume dependencies are handled by 'uv' inside the repo
        _run(cmd, cwd=repo_dir)
    except Exception as e:
        print(f"Run failed or interrupted: {e}")
        raise
    finally:
        vol.commit()
        print(f"Artifacts synced to {RUNS_DIR}")


@app.function(
    image=image,
    gpu=GPU_CONFIG, # Uses the big GPUs even for test to verify memory/setup
    timeout=20 * 60, # 20 mins max for test
    volumes={str(VOL_PATH): vol}
)
def test_10_steps(
    model: str = "d12",
    repo_ref: str = "master"
):
    """
    SMOKE TEST: Runs for 10 iterations only, saves, and exits.
    Use this to verify the 8x B200 setup works before doing a full run.
    """
    repo_dir = _setup_workspace(model, "https://github.com/karpathy/nanochat.git", repo_ref)
    
    print("\n>>> CONFIGURING FOR SMOKE TEST (10 Steps) <<<\n")
    
    # 1. Modify config files in place to force short run
    # Find the config file (e.g., config/d12.py or config/train_gpt2.py)
    # We append a max_iters override to the config command
    
    # We'll invoke train.py manually for the test to strictly control it
    # First, ensure data is prepared (speedrun.sh usually does this first)
    # If data doesn't exist, we run the prep step only.
    if not (repo_dir / "data" / "fineweb").exists():
         print("Preparing data for test...")
         _run("bash -c 'source speedrun.sh && prepare'", cwd=repo_dir)

    print("Running training for 10 steps...")
    # Override: max_iters=10, save_interval=5, eval_interval=10
    cmd = (
        f"uv run train.py config/{model}.py "
        "--max_iters=10 "
        "--log_interval=1 "
        "--eval_interval=10 "
        "--always_save_checkpoint=True"
    )
    
    _run(cmd, cwd=repo_dir)
    
    # Verify checkpoint exists
    ckpt_path = RUNS_DIR / model / "ckpt.pt"
    if ckpt_path.exists():
        print(f"\nSUCCESS: Checkpoint created at {ckpt_path}")
    else:
        print("\nWARNING: No checkpoint found after test run.")
    
    vol.commit()

