from os import path
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

MODEL = 'lstm'
EXPERIMENT_NAME = f"/archimedes-dl-{MODEL}"

client = MlflowClient()
experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

runs = mlflow.search_runs(
    experiment_ids=[ experiment_id ],
    #filter_string=f"attribute.status='{status}'",
    output_format="pandas",
    order_by=["metrics.start_time DESC"],
)
last_run = runs.iloc[0]

input_done = ["radius_20220918","residues_rolling_20220918", "radius_rolling_20220918", "residues_20220918"]

input_tables = ["radius_windowing_20220918", "radius_windowing__1_20220918", 
"residues_windowing_20220918", "residues_windowing__1_20220918"]

label_tables = ["levels_20220918", "labels_20220918"]

#input_tables = ["BIODARW_20220918"]
#label_tables = ["labels_20220918"]

for features_table in input_tables:
    for labels_table in label_tables:
        for n_classes in range(1,5,2):
            for n_layers in range(0,6):
                for n_units in range(3, 10):
                    for drop_out in np.arange(0, 0.5, 0.25):
                        params = {"test_ratio": 0.33, "n_units": 2**n_units,
                                    "n_layers": n_layers, "drop_out": drop_out,
                                    "mini_batch_size": 4 if "window" not in features_table else 256, "seed": 38,
                                    "max_epoch": 15000, "features_table": features_table, "labels_table": labels_table}
                        mlflow.run(".", parameters=params, experiment_id=experiment_id)
