from os import path
import numpy as np
import mlflow

experiment_name = "/archimedes-fcnn"

source_dir = "/data/elekin/data/results/handwriting/tmp"

files = [
    path.join(source_dir, "residues_17_20220912.csv"),
    path.join(source_dir, "radius_20220912.csv"),
    path.join(source_dir, "rolling_radius_std__20220912.csv"),
    path.join(source_dir, "rolling_residues_std_17_20220912.csv"),
    path.join(source_dir,'windowed_data_augmentation_rolling_residues_17_20220912.csv'),
    path.join(source_dir,'windowed_data_augmentation_residues_17_20220912.csv'),
    path.join(source_dir,'windowed_data_augmentation_radius__20220912.csv'),
    path.join(source_dir,'windowed_data_augmentation_rolling_radius__20220912.csv')]

models = ["tiny", "small", "large"]

for file in sorted(files):
    for n_classes in [3, 1]:
        for model in models: 
                    for drop_out in np.arange(0, 0.5, 0.25):
                        params = {"test_ratio": 0.33, "drop_out": drop_out,
                                    "mini_batch_size": 4 if "window" not in file else 128, "seed": 38,
                                    "max_epoch": 15000, "n_classes": n_classes,
                                    "filepath": file, "model":model}
                        mlflow.run(".", parameters=params, experiment_name=experiment_name)
