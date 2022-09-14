from os import path
import numpy as np
import mlflow

experiment_name = "/archimedes-fcnn"

source_dir = "/data/elekin/data/results/handwriting/tmp"

done_files = [path.join(source_dir, "residues_17_20220912.csv")]

files = [
    path.join(source_dir, "radius_20220912.csv"),
    path.join(source_dir, "rolling_radius_std__20220912.csv"),
    path.join(source_dir, "rolling_residues_std_17_20220912.csv"),
    path.join(source_dir,'windowed_data_augmentation_rolling_residues_17_20220912.csv'),
    path.join(source_dir,'windowed_data_augmentation_residues_17_20220912.csv'),
    path.join(source_dir,'windowed_data_augmentation_radius__20220912.csv'),
    path.join(source_dir,'windowed_data_augmentation_rolling_radius__20220912.csv')]

for file in sorted(files):
    for n_classes in range(1,5,2):
        for mini_batch_size in [4] if "window" not in file else [128]:#Fails with the whole dataset size 53 and 512 units 
            for n_layers in range(0,6):
                for n_units in range(1, 10):
                    for drop_out in np.arange(0, 0.5, 0.25):
                        params = {"test_ratio": 0.33, "n_units": n_units**2,
                                    "n_layers": n_layers, "drop_out": drop_out,
                                    "mini_batch_size": mini_batch_size, "seed": 38,
                                    "max_epoch": 5000, "n_classes": n_classes,
                                    "filepath": file}
                        mlflow.run(".", parameters=params, experiment_name=experiment_name)
