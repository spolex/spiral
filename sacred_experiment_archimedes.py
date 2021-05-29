from sacred import Experiment
import pandas as pd
from os import path

ex = Experiment()
ex.add_config("conf/sacred_config.json")

@ex.config
def archimedes_config(data_dir, coefficients):
    h5filename = path.join(data_dir,"archimedean-"+str(coefficients)+".h5")

@ex.capture
def get_h5_filename(data_dir, coefficients):
    return path.join(data_dir,"archimedean-"+str(coefficients)+".h5")

@ex.automain
def archimedes_experiment():
    HDFStore = pd.HDFStore
    h5filename = get_h5_filename()
    hdf = HDFStore(h5filename)
    print(hdf.keys())
    hdf.close()