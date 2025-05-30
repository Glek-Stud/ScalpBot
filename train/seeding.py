# W:\Jupiter\Thesis\train\seeding.py

import os
# must come before any TF import:
os.environ["TF_ENABLE_ONEDNN_OPTS"]   = "0"
os.environ["TF_DETERMINISTIC_OPS"]    = "1"

import fwr13y.d9m.tensorflow as fr_tf
fr_tf.enable_determinism()

import tensorflow as tf
import random

import numpy as np

def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    # NumPy
    np.random.seed(seed)

    # TensorFlow (eager + graph)
    tf.random.set_seed(seed)

    # Optional: deterministic GPU kernels (may slow CuDNN a little)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

