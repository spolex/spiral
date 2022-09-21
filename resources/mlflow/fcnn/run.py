from os import path
import numpy as np
import mlflow

experiment_name = "/archimedes-dl-fcnn"

input_tables = ["radius_20220918","radius_rolling_20220918", "radius_windowing_20220918", "radius_windowing__1_20220918", "residues_20220918","residues_rolling_20220918", "residues_windowing_20220918", "residues_windowing__1_20220918"]
label_tables = ["levels_20220918", "labels_20220918"]

#input_tables = ["BIODARW_20220918"]
#label_tables = ["labels_20220918"]

models = ["tiny", "small", "large"]

for features_table in sorted(input_tables):
    for labels_table in sorted(label_tables):
        for model in models: 
                    for drop_out in np.arange(0, 0.5, 0.25):
                        params = {"test_ratio": 0.33, "drop_out": drop_out, "n_classes":3 if "level" in labels_table else 1, 
                                    "mini_batch_size": 4 if "window" not in features_table else 128, "seed": 38,
                                    "max_epoch": 15000, "features_table": features_table, "labels_table": labels_table,"model":model}
                        mlflow.run(".", parameters=params, experiment_name=experiment_name)
