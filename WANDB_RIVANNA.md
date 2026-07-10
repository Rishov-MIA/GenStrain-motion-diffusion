# Weights & Biases on UVA Rivanna — step by step

Watch training-loss curves live at https://wandb.ai. Your API key lives in
`wandb.env` (git-ignored — never committed).

Your Rivanna working dir is `/scratch/vst2hb/GenStrain-motion-diffusion/`, so
all paths below are relative to there.

## One-time setup

1. **Make a wandb account** at https://wandb.ai (free for academic use).

2. **Copy your API key** from https://wandb.ai/authorize.

3. **Install wandb into the project env** (on a Rivanna login node):
   ```bash
   source activate video-diffusion-env
   pip install wandb                # already in requirements.txt
   ```

4. **Put your key on Rivanna.** Copy `wandb.env` into your working dir and add
   your key:
   ```bash
   cd /scratch/vst2hb/GenStrain-motion-diffusion/
   # edit wandb.env and replace PASTE_YOUR_KEY_HERE with your real key
   ```
   `wandb.env` must sit in the working dir the job `cd`s into, because the
   SLURM script does `source wandb.env` after `cd`.
   (Alternative: run `wandb login` once on the login node — it saves the key to
   `~/.netrc` and the job picks it up even without `wandb.env`.)

5. **Turn logging on** in the config you'll run. In its `"logging"` block set:
   ```json
   "logging": {
       "use_wandb": true,
       "wandb_project": "genstrain-motion-diffusion"
   }
   ```

6. **Set your allocation** in the SLURM script: edit the
   `#SBATCH --account=<>` line in `slurm/train_wandb.slurm` to your allocation
   name.

## Submitting a job

```bash
cd /scratch/vst2hb/GenStrain-motion-diffusion/
sbatch slurm/train_wandb.slurm
```

The script activates the env, sources `wandb.env`, and runs
`python train_full_region_augmented.py --aug 5` (edit that last line for a
different script/config). Monitor it:

```bash
squeue -u $USER                                   # job status
tail -f rivanna_logs/aug_<jobid>.log              # console log + wandb run URL
```

Click the `wandb: View run at https://wandb.ai/...` line in the log to open the
live dashboard.

## Multi-GPU (DDP) jobs

For the `*_ddp.py` scripts, request more GPUs (`--gres=gpu:a100:2`) and launch
with `torchrun` instead of `python`. Change the last line to:

```bash
torchrun --standalone --nproc_per_node=2 train_full_region_ddp.py
```

Only rank 0 logs to wandb, so you still get one clean run per job.

## If a compute node has no internet (offline mode)

Rivanna GPU nodes usually have outbound internet, so online mode normally works.
If a job can't reach wandb, set `export WANDB_MODE=offline` in `wandb.env`,
rerun, then upload afterward from a **login node**:

```bash
cd /scratch/vst2hb/GenStrain-motion-diffusion/
wandb sync wandb/offline-run-*
```

## Where do experiments show up?

Each config's `experiment_name` becomes a separate **run** under the
`genstrain-motion-diffusion` **project**. Different experiment → different run
(compare side by side); resuming the same experiment continues its run.

## What gets logged

- All trainers: `loss` (raw per step) and `loss_cummean` (cumulative mean since
  start — a smooth trend line).
- Group-DRO also: `weighted_loss`, `dro_score`, `dro_q`, per-group weights
  `q_<group>`, and per-group loss curves `loss_<group>` / `loss_<group>_cummean`.

See `configs/README.md` for the full field reference.
