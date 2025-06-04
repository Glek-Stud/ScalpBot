# train/run.py ─ one-click entry-point for train / eval
from __future__ import annotations

from __future__ import annotations
import argparse
import sys                         # already used for tensorboard call
import json                        # ← NEW
from dataclasses import asdict     # ← NEW
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
    p.add_argument(
        "--cfg",
        default="env_binance_tier0",
        help="name of env YAML inside configs/")

    return p.parse_args()


# ───────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()
    set_global_seed(args.seed)


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
        cfg_name=args.cfg,
        dueling=args.dueling,      # <-- the switch is forwarded here
    )

    if args.mode == "train":
        trainer.train(max_steps=400_000)

        # ── (2) BACK-TEST on test slice & save equity CSV ─────────
        metrics = trainer.evaluate("test", save_dir="reports")
        print("Test metrics:", metrics)  # console summary

        # ── (3) SAVE hyper-params & env YAML to /reports ----------
        from pathlib import Path, shutil
        import json
        Path("reports").mkdir(exist_ok=True)
        with open("reports/final_params.json", "w") as f:
            json.dump(asdict(trainer.params), f, indent=2)
        shutil.copy(Path("configs") / args.cfg,
                    "reports/env_used.yaml")

        # ── (4) EXPORT learning curves from TensorBoard -----------
        import subprocess, sys
        subprocess.run([
            sys.executable, "-m", "tensorboard", "dataexport",
            "--logdir", str(args.logdir),
            "--scalars", "regex=train/loss|val/Sharpe",
            "--out_format", "csv",
        ], stdout=open("reports/learning_curves.csv", "w"))

        # ── (5) SAVE model in .keras format -----------------------
        trainer.online.save("reports/dqn_best.keras",
                            include_optimizer=False)



    else:  # eval / back-test
        # load best checkpoint then evaluate on test slice
        ckpt = args.logdir / "checkpoints" / "dqn_best.h5"
        trainer.online.load_weights(ckpt)
        metrics = trainer.evaluate("test")
        print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
