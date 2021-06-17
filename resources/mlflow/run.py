import mlflow

experiment_name = "/archimedes-dl"

# Run MLflow project and create a reproducible conda environment
# on a local host
for n_layers in [2, 3, 5]:
    for n_units in [32, 64, 128, 256, 512]:
        for drop_out in [0.75, 0.5, 0.25]:
            for test_ratio in [0.15, 0.33, 0.5]:
                params = {"test_ratio": test_ratio, "n_units": n_units, "n_layers": n_layers, "drop_out": drop_out}
                mlflow.run(".", parameters=params, experiment_name=experiment_name)
