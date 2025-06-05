import argparse, json
from train.trainer import DQNTrainer, TrainerParams

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--logdir", default="runs/backtest")
args = parser.parse_args()

trainer = DQNTrainer(seed=42, logdir=args.logdir,
                     params=TrainerParams(), dueling=True)
trainer.online.load_weights(args.ckpt)
metrics = trainer.evaluate(split="test")
print(json.dumps(metrics, indent=2))