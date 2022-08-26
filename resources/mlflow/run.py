import mlflow

experiment_name = "/archimedes-lstm-xy"

# Run MLflow project and create a reproducible conda environment
# on a local host
for mini_batch_size in [4]:
    for n_layers in [0, 2, 3]:
        for n_units in [32, 64, 128, 256]:
            for drop_out in [0.0, 0.2]:
                params = {"test_ratio": 0.33, "n_units": n_units,
                            "n_layers": n_layers, "drop_out": drop_out,
                            "mini_batch_size": mini_batch_size, "seed": 38,
                            "max_epoch": 1000}
                mlflow.run(".", parameters=params, experiment_name=experiment_name)
