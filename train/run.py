# W:\Jupiter\Thesis\train\run.py
# run.py  ─ one-click entry-point for train / eval

from __future__ import annotations
import argparse
import tensorflow as tf

from pathlib import Path
from .seeding import set_global_seed
from .model import build_q_network, hard_update
from .trainer import DQNTrainer, TrainerParams


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["train", "eval"], default="train",
                   help="train = learn + val  |  eval = load best ckpt and back-test")
    p.add_argument("--seed", type=int, default=42, help="global RNG seed")
    p.add_argument("--logdir", type=Path, default=Path("runs/tensorboard"),
                   help="TensorBoard output directory")
    # (later: --cfg, --checkpoint, etc.)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_global_seed(args.seed)


    if args.mode == "train":
        trainer = DQNTrainer(seed=args.seed,
                             logdir=args.logdir,
                             params=TrainerParams())
        trainer.train(max_steps=300_000)


    else:  # eval / back-test
        trainer = DQNTrainer(seed=args.seed,
                             logdir=args.logdir,
                             params=TrainerParams())
        # load best checkpoint before evaluate
        trainer.online.load_weights(args.logdir / "checkpoints" / "dqn_best.h5")
        metrics = trainer.evaluate("test")
        print("Test metrics:", metrics)




    args = _parse_args()
    set_global_seed(args.seed)




if __name__ == "__main__":
    main()
