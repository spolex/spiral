from os import path
import numpy as np
import mlflow

experiment_name = "/archimedes-dl-lstm"

source_dir = "/data/elekin/data/results/handwriting/tmp"

input_tables = ["radius_20220918","radius_rolling_20220918", "radius_windowing_20220918", "radius_windowing__1_20220918", "residues_20220918","residues_rolling_20220918", "residues_windowing_20220918", "residues_windowing__1_20220918"]
label_tables = ["levels_20220918", "labels_20220918"]

#input_tables = ["BIODARW_20220918"]
#label_tables = ["labels_20220918"]

for features_table in sorted(input_tables):
    for labels_table in sorted(label_tables):
        for n_classes in range(1,5,2):
            for n_layers in range(0,6):
                for n_units in range(1, 10):
                    for drop_out in np.arange(0, 0.5, 0.25):
                        params = {"test_ratio": 0.33, "n_units": n_units**2,
                                    "n_layers": n_layers, "drop_out": drop_out,
                                    "mini_batch_size": 4 if "window" not in features_table else 128, "seed": 38,
                                    "max_epoch": 15000, "features_table": features_table, "labels_table": labels_table}
                        mlflow.run(".", parameters=params, experiment_name=experiment_name)
