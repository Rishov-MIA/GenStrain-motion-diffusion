import copy
import inspect
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.utils import data

from video_diffusion_pytorch.utils import generate_displacement_quiver_gifs_with_gt
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    EMA,
    Trainer,
    exists,
    noop,
    normalize_cond_img,
    random_pick_condition_videos,
)


# This order is also the string-match priority.
ALLOWED_DISEASE_GROUPS = (
    "LBBB",
    "Healthy Pediatric",
    "Acute MI",
    "Myocarditis",
    "DCM",
    "Healthy",
)


def disease_to_group(disease, allowed_groups=ALLOWED_DISEASE_GROUPS):
    if not isinstance(disease, str):
        return None

    disease_lower = disease.lower()
    for group_name in allowed_groups:
        if group_name.lower() in disease_lower:
            return group_name
    return None


class GroupedContourDataset(data.Dataset):
    def __init__(
        self,
        input_video_folder,
        image_size,
        contour_condition_video_dir,
        metadata_json_path,
        channels=2,
        num_frames=16,
        exts=("npy",),
        allowed_groups=ALLOWED_DISEASE_GROUPS,
    ):
        super().__init__()
        self.input_video_folder = Path(input_video_folder)
        self.contour_condition_video_dir = Path(contour_condition_video_dir)
        self.metadata_json_path = Path(metadata_json_path)
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        self.allowed_groups = tuple(allowed_groups)
        self.cache = {}

        if not self.metadata_json_path.exists():
            raise FileNotFoundError(f"metadata JSON not found: {self.metadata_json_path}")

        with self.metadata_json_path.open("r") as f:
            metadata = json.load(f)

        input_paths = [p for ext in exts for p in self.input_video_folder.glob(f"**/*.{ext}")]
        contour_paths = [p for ext in exts for p in self.contour_condition_video_dir.glob(f"**/*.{ext}")]

        input_by_name = {p.name: p for p in input_paths}
        contour_by_name = {p.name: p for p in contour_paths}
        input_names = set(input_by_name)
        contour_names = set(contour_by_name)
        missing_contour = sorted(input_names - contour_names)
        missing_input = sorted(contour_names - input_names)
        assert not missing_contour and not missing_input, (
            "Target and contour folders must contain the same filenames. "
            f"Missing contour files for {len(missing_contour)} targets; "
            f"missing target files for {len(missing_input)} contours."
        )
        paired_names = sorted(input_names)

        skipped = Counter()
        raw_samples = []

        for filename in paired_names:
            info = self._lookup_metadata(metadata, filename)
            if info is None:
                skipped["missing_metadata"] += 1
                continue

            group_name = disease_to_group(info.get("disease"), self.allowed_groups)
            if group_name is None or group_name not in self.allowed_groups:
                skipped["unsupported_disease"] += 1
                continue

            raw_samples.append(
                {
                    "input_path": input_by_name[filename],
                    "contour_path": contour_by_name[filename],
                    "filename": filename,
                    "group_name": group_name,
                }
            )

        self.group_names = [group for group in self.allowed_groups if any(s["group_name"] == group for s in raw_samples)]
        self.group_to_idx = {group_name: idx for idx, group_name in enumerate(self.group_names)}
        self.samples = []
        self.group_to_indices = defaultdict(list)

        for sample in raw_samples:
            group_idx = self.group_to_idx[sample["group_name"]]
            sample["group_idx"] = group_idx
            self.group_to_indices[group_idx].append(len(self.samples))
            self.samples.append(sample)

        self.group_counts = [len(self.group_to_indices[idx]) for idx in range(len(self.group_names))]
        self.skipped = dict(skipped)

    @staticmethod
    def _lookup_metadata(metadata, filename):
        path = Path(filename)
        for key in (filename, f"{path.stem}.mat", path.stem):
            if key in metadata:
                return metadata[key]
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        if index not in self.cache:
            input_video_tensor = torch.from_numpy(np.load(sample["input_path"])).float()
            contour_condition_video_tensor = torch.from_numpy(np.load(sample["contour_path"])).float()
            self.cache[index] = (input_video_tensor, contour_condition_video_tensor)
        else:
            input_video_tensor, contour_condition_video_tensor = self.cache[index]

        return (
            input_video_tensor.squeeze(0),
            contour_condition_video_tensor,
            sample["group_idx"],
            sample["filename"],
        )

    def sample_group_batch(self, group_idx, batch_size):
        indices = self.group_to_indices[group_idx]
        assert len(indices) > 0, f"group {group_idx} has no samples"

        chosen = [indices[random.randrange(len(indices))] for _ in range(batch_size)]
        input_videos = []
        contour_condition_videos = []
        filenames = []

        for index in chosen:
            input_video, contour_condition_video, _, filename = self[index]
            input_videos.append(input_video)
            contour_condition_videos.append(contour_condition_video)
            filenames.append(filename)

        return torch.stack(input_videos, dim=0), torch.stack(contour_condition_videos, dim=0), filenames


class GroupDROTrainer(Trainer):
    def __init__(
        self,
        diffusion_model,
        input_video_folder,
        *,
        contour_condition_video_dir=None,
        sampling_contour_condition_video_dir=None,
        metadata_json_path=None,
        dro_eta_q=0.01,
        dro_adjustment_c=0.0,
        weight_decay=0.0,
        ema_decay=0.995,
        num_frames=16,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=1,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        max_grad_norm=None,
        experiment_name="test-exp",
        inference_only=False,
    ):
        object.__init__(self)
        if gradient_accumulate_every != 1:
            raise ValueError("GroupDROTrainer currently supports gradient_accumulate_every=1")
        if metadata_json_path is None and not inference_only:
            raise ValueError("metadata_json_path is required for GroupDROTrainer")

        self.model = diffusion_model
        self.device = next(diffusion_model.parameters()).device
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.sampling_contour_condition_video_dir = sampling_contour_condition_video_dir

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.dro_eta_q = float(dro_eta_q)
        self.dro_adjustment_c = float(dro_adjustment_c)
        self.weight_decay = float(weight_decay)

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        if not inference_only:
            self.ds = GroupedContourDataset(
                input_video_folder,
                image_size,
                contour_condition_video_dir,
                metadata_json_path,
                channels=channels,
                num_frames=num_frames,
            )

            print(f"found {len(self.ds)} supported grouped videos as .npy files at {input_video_folder}")
            print(f"group counts: {dict(zip(self.ds.group_names, self.ds.group_counts))}")
            if self.ds.skipped:
                print(f"skipped files: {self.ds.skipped}")

            assert len(self.ds) > 0, "need at least 1 supported grouped video to start training"
            assert len(self.ds.group_names) > 0, "need at least 1 active group to start training"

            self.group_names = list(self.ds.group_names)
            self.group_counts = torch.tensor(self.ds.group_counts, dtype=torch.float32, device=self.device)
            self.dro_log_q = torch.full(
                (len(self.group_names),),
                -math.log(len(self.group_names)),
                dtype=torch.float32,
                device=self.device,
            )
            self.opt = Adam(diffusion_model.parameters(), lr=train_lr, weight_decay=self.weight_decay)
        else:
            self.group_names = []
            self.group_counts = torch.empty(0, dtype=torch.float32, device=self.device)
            self.dro_log_q = torch.empty(0, dtype=torch.float32, device=self.device)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.experiment_name = experiment_name

        checkpoints_folder = f"./{self.experiment_name}/checkpoints"
        self.checkpoints_folder = Path(checkpoints_folder)
        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    @property
    def dro_q(self):
        return self.dro_log_q.exp()

    def save(self, milestone):
        ckpt = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "scaler": self.scaler.state_dict(),
            "dro_log_q": self.dro_log_q.detach().cpu(),
            "dro_group_names": list(self.group_names),
            "dro_group_counts": self.group_counts.detach().cpu(),
            "dro_eta_q": self.dro_eta_q,
            "dro_adjustment_c": self.dro_adjustment_c,
            "dro_weight_decay": self.weight_decay,
        }
        torch.save(ckpt, str(self.checkpoints_folder / f"model-{milestone}.pt"))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split("-")[-1]) for p in Path(self.checkpoints_folder).glob("**/*.pt")]
            assert len(all_milestones) > 0, "need at least one milestone to load from (milestone == -1)"
            milestone = max(all_milestones)

        ckpt_path = str(self.checkpoints_folder / f"model-{milestone}.pt")
        if "weights_only" in inspect.signature(torch.load).parameters:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        else:
            ckpt = torch.load(ckpt_path, map_location=self.device)

        print(f"loaded checkpoint: model-{milestone}.pt\n")

        self.step = ckpt["step"]
        self.model.load_state_dict(ckpt["model"], **kwargs)
        self.ema_model.load_state_dict(ckpt["ema"], **kwargs)
        self.scaler.load_state_dict(ckpt["scaler"])

        if "dro_log_q" not in ckpt:
            print("checkpoint has no DRO state; keeping uniform group weights")
            return

        checkpoint_group_names = list(ckpt.get("dro_group_names", []))
        if checkpoint_group_names != self.group_names:
            raise ValueError(
                "DRO checkpoint group names do not match the current dataset: "
                f"checkpoint={checkpoint_group_names}, current={self.group_names}"
            )

        self.dro_log_q = ckpt["dro_log_q"].to(self.device, dtype=torch.float32)

    def sample_group_idx(self):
        return random.randrange(len(self.group_names))

    def update_dro_weight(self, group_idx, group_loss):
        with torch.no_grad():
            adjustment = self.dro_adjustment_c / self.group_counts[group_idx].sqrt()
            score = group_loss.detach().float() + adjustment
            self.dro_log_q[group_idx] = self.dro_log_q[group_idx] + self.dro_eta_q * score
            self.dro_log_q = self.dro_log_q - torch.logsumexp(self.dro_log_q, dim=0)
            q_g = self.dro_q[group_idx].detach()
        return score.detach(), q_g

    def train(self, prob_focus_present=0.0, focus_present_mask=None, log_fn=noop):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            group_idx = self.sample_group_idx()
            group_name = self.group_names[group_idx]

            input_video, contour_cond_video, _ = self.ds.sample_group_batch(group_idx, self.batch_size)
            input_video = input_video.to(self.device, non_blocking=True)
            contour_cond_video = contour_cond_video.to(self.device, non_blocking=True)

            with autocast(enabled=self.amp):
                loss, x0, x_start = self.model(
                    input_video,
                    cond=[contour_cond_video],
                    prob_focus_present=prob_focus_present,
                    focus_present_mask=focus_present_mask,
                )

            score, q_g = self.update_dro_weight(group_idx, loss)
            weighted_loss = q_g * loss
            self.scaler.scale(weighted_loss).backward()

            log = {
                "loss": loss.item(),
                "weighted_loss": weighted_loss.item(),
                "dro_score": score.item(),
                "dro_q": q_g.item(),
                "group": group_name,
            }

            q_parts = ", ".join(f"{name}={q:.4f}" for name, q in zip(self.group_names, self.dro_q.detach().cpu()))
            print(
                f"{self.step}: group={group_name} "
                f"loss={loss.item():.6f} weighted={weighted_loss.item():.6f} q={q_g.item():.6f} | {q_parts}"
            )

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if (self.step % 300 == 0) and (self.step != 0):
                predicted_start_training_dir = f"./{self.experiment_name}/predicted_start_training_gifs"
                os.makedirs(predicted_start_training_dir, exist_ok=True)
                x0_np = x0.detach().cpu().numpy()
                x_start_np = x_start.detach().cpu().numpy()

                generate_displacement_quiver_gifs_with_gt(
                    milestone="train",
                    filenames=[f"train_visual_{self.step}_{group_name}"],
                    pred_disp=np.expand_dims(x0_np[0:1], axis=0),
                    gt_disp=np.expand_dims(x_start_np[0:1], axis=0),
                    output_dir=predicted_start_training_dir,
                )

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

                sample_contour_cond, sample_cond_filenames = random_pick_condition_videos(
                    num_samples=1,
                    contour_cond_video_dir=self.sampling_contour_condition_video_dir,
                )
                sample_contour_cond = normalize_cond_img(sample_contour_cond)
                sample_contour_cond = sample_contour_cond.to(self.device)

                self.sample_and_save(
                    milestone,
                    contour_cond_video=sample_contour_cond,
                    cond_filenames=sample_cond_filenames,
                    save_folder=f"./{self.experiment_name}/sampled_videos_infos",
                )

            log_fn(log)
            self.step += 1

        print("training completed")
