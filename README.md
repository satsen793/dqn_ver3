# DQN Training for Adaptive Learning Simulator

This repo contains a self-contained simulator and a baseline DQN agent with prioritized replay. See the spec_*.md files for design details.

## Requirements
- Windows 64-bit
- Python 3.10 (recommended for wheel availability)

## Setup (Local Windows)
```powershell
# From the repo root
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Training (Local)
Default run (500 episodes, 5000 warmup steps, max 140 steps/episode):
```powershell
python train_dqn.py
```
Your requested config (seed=0, 200 steps/episode):
```powershell
python train_dqn.py --seed 0 --steps 200
```
Quick smoke test (3 episodes, no warmup):
```powershell
python train_dqn.py --seed 0 --steps 200 --episodes 3 --start-steps 0
```

## HPC Run (SLURM)

### 1) Create environment (CPU default)
```bash
conda env create -f environment.yml
conda activate dqn_ver3
```

If your cluster doesn’t ship CPU PyTorch via conda, install via pip:
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install numpy
```

For NVIDIA GPU nodes, switch the CUDA build per your cluster’s docs (update `environment.yml` or use the PyTorch selector). In `run_dqn.slurm`, set your GPU partition and uncomment `--gres=gpu:1`.

### 2) Smoke runs (HPC)

- Simulator validation (expected: "✓ Simulator validation checks passed"):
```bash
srun -p cpu --time=00:10:00 --mem=4G \
	bash -lc "conda activate dqn_ver3 && python -c 'from train_dqn import validate_simulator; validate_simulator()'"
```

- Single-episode functional check (expected: prints "Training finished."):
```bash
srun -p cpu --time=00:20:00 --mem=4G \
	bash -lc "conda activate dqn_ver3 && python -c 'from train_dqn import run_training; out=run_training(num_episodes=1, max_steps_per_episode=100, start_steps=0, seed=0); print("ok", len(out["returns"]))'"
```

- CLI smoke (3 episodes, no warmup) — expected: "Training finished."; no 10-episode summaries printed:
```bash
srun -p cpu --time=00:30:00 --mem=4G \
	bash -lc "conda activate dqn_ver3 && python train_dqn.py --seed 0 --steps 120 --episodes 3 --start-steps 0"
```

### 3) Standard single-seed job
Submit SLURM job; logs will appear under `logs/`.
```bash
sbatch run_dqn.slurm
```
Override parameters at submit time:
```bash
SEED=0 STEPS=200 EPISODES=500 START_STEPS=5000 sbatch run_dqn.slurm
```

Expected outputs in `logs/dqn_<jobid>.out`:
- Every 10 episodes: lines like `Episode 0010 | return=XX.XX | ttm=140.0 | steps=YYYY`
- Final: `Training finished.`

### 4) Multi-seed summary job (recommended for evaluation)
Submit the multi-seed SLURM script; a JSON summary is saved to `logs/hpc_summary.json`.
```bash
SEEDS=0,1,2,3,4 EPISODES=200 STEPS=140 START_STEPS=5000 sbatch run_multi.slurm
```

Expected outputs:
- In `logs/dqn_multi_<jobid>.out`: a printed summary block like
	- `Summary (across seeds):`
	- `- cumulative_reward: {"mean": ..., "sd": ..., "ci_lower": ..., "ci_upper": ...}`
	- `- time_to_mastery: {"mean": ..., "sd": ..., "ci_lower": ..., "ci_upper": ..., "median": ..., "p25": ..., "p75": ...}`
	- `- blueprint_adherence: {...}`
	- `- post_content_gain: {...}`
	- `- policy_stability: {"sd": ..., "cv": ..., "ci_lower": ..., "ci_upper": ...}`
- JSON file written to `logs/hpc_summary.json` with full details.
- Combined CSV written to `logs/hpc_summary.csv` containing per-episode metrics across all seeds.
	- Columns include per-modality post-content gains: video, PPT, text, blog, article, handout.

To change output paths:
```bash
OUT=logs/my_summary.json OUT_CSV=logs/my_summary.csv SEEDS=0,1,2,3,4 sbatch run_multi.slurm
```

## Notes
- Torch can be installed from the CPU-only index (see `requirements.txt`). For CUDA builds, follow the official selector at https://pytorch.org/get-started/locally/ and adjust installation accordingly.
- `scipy` is only required for the optional `compare_algorithms()` function. Install it if you plan to run that analysis:
```powershell
pip install scipy
```
- If wheels fail to resolve on newer Python versions, use Python 3.10 as shown above.

## Blueprint Compliance Checklist
- State: 30 mastery + frustration + response time = 32 dims.
- Actions: 90 question (30×3) + 180 content (30×6) = 270.
- IRT ranges: a∈[0.5,2.0], b∈[-2.0,2.0], c∈[0.1,0.25] by difficulty bucket.
- Reward weights: correctness=1.0, mastery_gain=0.5, frustration_penalty=0.3, post_content_gain=2.0, engagement_bonus=0.5.
- Episode termination: mastery ≥ 0.8, step budget default 140, critical frustration ≥ 0.95.
- Fail-streak gate: at ≥3, redirect question to content-video.
- Blueprint mix: strict difficulty masking in exploration and greedy action selection.

## Optional: Per-episode CSV/JSON on single runs
You can dump per-episode metrics with the single-run CLI:
```bash
python train_dqn.py --seed 0 --steps 140 --episodes 50 --out-csv logs/single_seed_ep.csv --out-json logs/single_seed_results.json
```
CSV columns: episode, return, ttm, total_steps, final_mastery, cumulative_reward, question_accuracy, content_rate, blueprint_adherence, post_content_gain, post_content_gain_video, post_content_gain_PPT, post_content_gain_text, post_content_gain_blog, post_content_gain_article, post_content_gain_handout, mean_frustration.
