"""Train and evaluate MXPose on Human3.6M."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.h36m_dataset import Human36mDataset, TEST_SUBJECTS, TRAIN_SUBJECTS
from common.metrics import ActionMetrics
from model import MXPose
from training.pmss import ProgressiveMultiStageSupervision, stage_for_epoch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--keypoints", default="cpn_ft_h36m_dbb")
    parser.add_argument("--keypoints-path", type=Path, help="Override the 2D NPZ path")
    parser.add_argument("--actions", default="*", help="Comma-separated actions, or *")
    parser.add_argument("--frames", type=int, default=27)
    parser.add_argument("--embed-dim", type=int, default=32, help="Channels per joint")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--k-neighbors", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-decay", type=float, default=0.98)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--stage1-end", type=int, default=35)
    parser.add_argument("--stage2-end", type=int, default=45)
    parser.add_argument("--bone-weight", type=float, default=0.9)
    parser.add_argument("--pose-weight", type=float, default=0.01)
    parser.add_argument("--stride", type=int, default=1, help="Training window-center stride")
    parser.add_argument("--downsample", type=int, default=1, help="Subsample source sequences")
    parser.add_argument("--reverse-augmentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evaluate", type=Path, help="Evaluate a saved MXPose checkpoint")
    mode.add_argument("--resume", type=Path, help="Resume training, including optimizer and PMSS")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-parallel", action="store_true", help="Use all visible CUDA devices")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.epochs < 1:
        parser.error("batch-size and epochs must be positive; workers must be nonnegative")
    if args.lr <= 0 or not 0 < args.lr_decay <= 1 or args.weight_decay < 0:
        parser.error("Require lr > 0, 0 < lr-decay <= 1 and weight-decay >= 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    if args.data_parallel and args.device != "cuda":
        parser.error("--data-parallel requires --device cuda; select GPUs with CUDA_VISIBLE_DEVICES")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def make_loader(args, train):
    dataset = Human36mDataset(
        args.data_root / "data_3d_h36m.npz",
        args.keypoints_path or args.data_root / f"data_2d_h36m_{args.keypoints}.npz",
        subjects=TRAIN_SUBJECTS if train else TEST_SUBJECTS,
        frames=args.frames, stride=args.stride if train else 1,
        downsample=args.downsample, actions=args.actions,
        reverse_augmentation=args.reverse_augmentation and train,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train,
                      num_workers=args.workers, pin_memory=args.device.startswith("cuda"))


def train_epoch(model, supervision, loader, optimizer, device, stage):
    model.train()
    supervision.train()
    totals, samples = {}, 0
    progress = tqdm(loader, desc=f"Train / PMSS {stage}", dynamic_ncols=True)
    for batch in progress:
        inputs = batch["input_2d"].to(device, non_blocking=True)
        targets = batch["target_3d"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        predicted = model(inputs, stage=stage)
        loss, components = supervision(predicted, targets, inputs, stage)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        count = len(inputs)
        samples += count
        for name, value in {"loss": loss, **components}.items():
            totals[name] = totals.get(name, 0.0) + value.detach().item() * count
        progress.set_postfix(loss=f"{totals['loss'] / samples:.5f}")
    return {name: total / samples for name, total in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, stage):
    model.eval()
    metrics = ActionMetrics()
    for batch in tqdm(loader, desc="Evaluate", dynamic_ncols=True):
        inputs = batch["input_2d"].to(device, non_blocking=True)
        predicted = model(inputs, stage=stage)
        center = inputs.shape[1] // 2
        predicted = predicted[:, center]
        predicted = predicted - predicted[:, :1]
        target = batch["target_3d"][:, center]
        metrics.update(predicted.cpu(), target, batch["action"])
    return metrics.compute()


def print_metrics(results):
    print(f"{'Action':<16} {'P1 (mm)':>10} {'P2 (mm)':>10}")
    for action, values in results.items():
        print(f"{action:<16} {values['mpjpe']:>10.2f} {values['p_mpjpe']:>10.2f}")


def save_checkpoint(path, state):
    """Atomically replace a completed checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    saved = None
    if args.evaluate or args.resume:
        # Checkpoints are locally trained artifacts containing optimizer/RNG state.
        saved = torch.load(args.evaluate or args.resume, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or "model_config" not in saved or "model" not in saved:
            raise ValueError("Expected a checkpoint produced by MXPose main.py")
        args.frames = saved["model_config"]["frames"]
        provided = {argument.split("=", 1)[0] for argument in sys.argv[1:] if argument.startswith("--")}
        # Keep detector/data sampling consistent with the saved experiment.
        # Explicit CLI options can override these for a deliberate new evaluation
        # or a different batch size; paths and device remain portable.
        restore_options = ["keypoints", "actions", "downsample"]
        if args.resume:
            restore_options += ["batch_size", "stride", "reverse_augmentation", "seed"]
        for name in restore_options:
            option = "--" + name.replace("_", "-")
            explicit = option in provided or (name == "reverse_augmentation" and "--no-reverse-augmentation" in provided)
            if not explicit:
                setattr(args, name, saved["args"][name])
        if saved["args"]["keypoints_path"] and not provided.intersection(
            {"--data-root", "--keypoints", "--keypoints-path"}
        ):
            args.keypoints_path = Path(saved["args"]["keypoints_path"])
    model_config = saved["model_config"] if saved else {
        "frames": args.frames, "embed_dim": args.embed_dim, "depth": args.depth,
        "num_heads": args.num_heads, "k_neighbors": args.k_neighbors,
    }
    for name, value in model_config.items():
        if hasattr(args, name):
            setattr(args, name, value)
    model = MXPose(**model_config).to(device)
    if saved:
        model.load_state_dict(saved["model"], strict=True)
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(f"MXPose: {sum(p.numel() for p in model.parameters()):,} parameters; {args.frames} frames; {device}")

    if args.evaluate:
        stage = saved["stage"]
        print(f"Checkpoint epoch {saved['epoch']}, PMSS stage {stage}")
        print_metrics(evaluate(model, make_loader(args, train=False), device, stage))
        return

    if args.resume:
        # Restore the curriculum and optimizer settings used by the original run.
        for name in ("stage1_end", "stage2_end", "bone_weight", "pose_weight",
                     "lr", "lr_decay", "weight_decay"):
            setattr(args, name, saved["args"][name])
    stage_for_epoch(1, args.stage1_end, args.stage2_end)
    supervision = ProgressiveMultiStageSupervision(args.bone_weight, args.pose_weight).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(supervision.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, amsgrad=True,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    start_epoch, best_p1 = 1, float("inf")
    if args.resume:
        supervision.load_state_dict(saved["supervision"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = saved["epoch"] + 1
        best_p1 = saved["best_p1"]
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        if device.type == "cuda" and len(saved["rng"]["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
    if start_epoch > args.epochs:
        raise ValueError(f"Checkpoint already reached epoch {start_epoch - 1}; increase --epochs")

    train_loader = make_loader(args, train=True)
    test_loader = make_loader(args, train=False)
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    (args.checkpoint / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for epoch in range(start_epoch, args.epochs + 1):
        stage = stage_for_epoch(epoch, args.stage1_end, args.stage2_end)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs}, PMSS stage {stage}, lr={learning_rate:.8f}")
        train_results = train_epoch(model, supervision, train_loader, optimizer, device, stage)
        results = evaluate(model, test_loader, device, stage)
        print_metrics(results)
        p1 = results["Average"]["mpjpe"]
        is_best = p1 < best_p1
        best_p1 = min(p1, best_p1)
        scheduler.step()
        state = {
            "epoch": epoch, "stage": stage, "model_config": model_config,
            "model": unwrap(model).state_dict(), "supervision": supervision.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_p1": best_p1, "args": config,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []},
        }
        save_checkpoint(args.checkpoint / "last.pth", state)
        if is_best:
            save_checkpoint(args.checkpoint / "best.pth", state)
        record = {"epoch": epoch, "stage": stage, "lr": learning_rate,
                  "train": train_results, "test": results, "best_p1": best_p1}
        with (args.checkpoint / "train.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
