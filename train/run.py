from __future__ import annotations
import argparse
import sys
import json
from dataclasses import asdict
from pathlib import Path

from .seeding import set_global_seed
from .trainer import DQNTrainer, TrainerParams


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

    p.add_argument(
        "--seed",
                   type=int,
                   default=42,
                   help="global RNG seed")

    p.add_argument(
        "--logdir",
        type=Path,
        default=Path("runs/tensorboard"),
        help="TensorBoard / checkpoint directory",
    )

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
    p.add_argument(
        "--cfg",
        default="env_binance_tier0",
        help="name of env YAML inside configs/")

    return p.parse_args()

def main() -> None:
    args = _parse_args()
    set_global_seed(args.seed)

    params = TrainerParams(
        lr=3e-4,
        batch_size=256,
        eps_decay=60_000,
        val_freq=10_000,
        patience=8,
        prioritised=True,
    )

    trainer = DQNTrainer(
        seed=args.seed,
        logdir=args.logdir,
        params=params,
        cfg_name=args.cfg,
        dueling=args.dueling,
    )

    if args.mode == "train":
        trainer.train(max_steps=400_000)

        metrics = trainer.evaluate("test", save_dir="reports")
        print("Test metrics:", metrics)

        from pathlib import Path, shutil
        import json
        Path("reports").mkdir(exist_ok=True)
        with open("reports/final_params.json", "w") as f:
            json.dump(asdict(trainer.params), f, indent=2)
        shutil.copy(Path("configs") / args.cfg,
                    "reports/env_used.yaml")

        import subprocess, sys
        subprocess.run([
            sys.executable, "-m", "tensorboard", "dataexport",
            "--logdir", str(args.logdir),
            "--scalars", "regex=train/loss|val/Sharpe",
            "--out_format", "csv",
        ], stdout=open("reports/learning_curves.csv", "w"))

        trainer.online.save("reports/dqn_best.keras",
                            include_optimizer=False)



    else:
        ckpt = args.logdir / "checkpoints" / "dqn_best.h5"
        trainer.online.load_weights(ckpt)
        metrics = trainer.evaluate("test")
        print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
