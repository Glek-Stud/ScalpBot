# train/run.py ─ one-click entry-point for train / eval
from __future__ import annotations

import json
import argparse
from pathlib import Path

from .seeding import set_global_seed
from .trainer import DQNTrainer, TrainerParams


# ───────────────────────────────────────────────────────────────
# CLI parser
# ───────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="train = learn + val | eval = back-test on test slice",
    )
    p.add_argument("--seed", type=int, default=42, help="global RNG seed")
    p.add_argument(
        "--logdir",
        type=Path,
        default=Path("runs/tensorboard"),
        help="TensorBoard / checkpoint directory",
    )

    # NEW: flag that toggles the dueling architecture
    p.add_argument(
        "--dueling",
        action="store_true",
        help="use dueling value-advantage network head",
    )

    p.add_argument(
        "--params",
        type=str,
        default="{}",
        help="JSON string of TrainerParams overrides "
             'e.g. --params \'{"lr":2e-4,"target_freq":250}\'',
    )

    return p.parse_args()


# ───────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()
    set_global_seed(args.seed)

    overrides = json.loads(args.params)
    params = TrainerParams(**overrides)

    # Common hyper-params (adjust as desired)
    params = TrainerParams(
        lr=3e-4,
        batch_size=256,
        eps_decay=60_000,
        val_freq=10_000,
        patience=8,
        prioritised=True,
    )

    # Instantiate trainer — pass dueling flag to choose architecture
    trainer = DQNTrainer(
        seed=args.seed,
        logdir=args.logdir,
        params=params,
        dueling=args.dueling,      # <-- the switch is forwarded here
    )

    if args.mode == "train":
        trainer.train(max_steps=400_000)

    else:  # eval / back-test
        # load best checkpoint then evaluate on test slice
        ckpt = args.logdir / "checkpoints" / "dqn_best.h5"
        trainer.online.load_weights(ckpt)
        metrics = trainer.evaluate("test")
        print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
