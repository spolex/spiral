from os import path
import yaml

with open('config.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    config = yaml.load(file, Loader=yaml.FullLoader)
    
    # loading data parameters
    num_coefficients=int(config["coefficients"])
    root_path=config["root_path"]
    hw_results_path= config["hw_results_path"]
    #Load data from hdf5 file
    rdo_root_path = path.join(root_path,hw_results_path)
    h5file = path.join(rdo_root_path, "archimedean-")
    h5filename = h5file + str(num_coefficients) + ".h5"
    h5_train_test_filename = h5file + str(num_coefficients) + "-splits.h5"

    # training parameters
    seed=int(config["seed"]) if "seed" in config.keys() else 42

    dr=float(config["dropout"]) if "dropout" in config.keys() else 0.2
    lr2=float(config["lr2"]) if "lr2" in config.keys() else 1e-3
    lr1=float(config["lr1"]) if "lr1" in config.keys() else 1e-4
    lr=float(config["lr"]) if "lr" in config.keys() else 8e-4

    num_epochs=int(config["num_epochs"]) if "num_epochs" in config.keys() else 1000
    num_features=int(config["features"]) if "features" in config.keys() else 4096
    mini_batch_size=int(config["mini_batch_size"]) if "mini_batch_size" in config.keys() else 4

    main_units=int(config["main_units"]) if "main_units" in config.keys() else 64
    secondary_units=int(config["secondary_units"]) if "secondary_units" in config.keys() else 16
    last_unit=int(config["last_unit"]) if "last_unit" in config.keys() else 8
    lstm_units=int(config["lstm_units"]) if "lstm_units" in config.keys() else 64
    num_classes=int(config["num_classes"]) if "num_classes" in config.keys() else 1
    
    #visualization parameters
    colors=["b", "r"]


    print_sample=False