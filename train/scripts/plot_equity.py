import pandas as pd, matplotlib.pyplot as plt, sys, pathlib as pl

csv_path = pl.Path(sys.argv[1])
df = pd.read_csv(csv_path)
plt.figure(figsize=(8,4))
plt.plot(df["equity"])
plt.title(f"Equity curve – {csv_path.stem}")
plt.xlabel("Step"); plt.ylabel("Equity")
plt.tight_layout()
plt.savefig(csv_path.with_suffix(".png"))
