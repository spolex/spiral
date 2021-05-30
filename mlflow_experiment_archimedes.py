from os import path

from sacred import Experiment
from sacred.observers import MongoObserver

import pandas as pd
import swifter

from properties.properties import Properties

from loaders.biodarw_loader import data_ingredient, load_dataset
from preprocess.biodarw_feature_extraction import extract_residues
from scipy.signal import resample

import seaborn as sns

print(Properties)

ex = Experiment('archimedes discrete cosine transform - DCT', ingredients=[data_ingredient])
ex.add_config("conf/sacred_config.json")
ex.observers.append(MongoObserver(url='mongodb://'+Properties["MONGO_INITDB_ROOT_USERNAME"]+':'+Properties["MONGO_INITDB_ROOT_PASSWORD"]+'@mongo:27017/admin', 
                                  db_name='archimedes'))

def calc_residues(dataset, coefficients):
    rd = dataset.groupby(Properties["subject_id"]).apply(extract_residues, c=coefficients)\
                    .swifter.apply(resample, num=Properties["resample"])
    
    return rd.apply(pd.Series)


@ex.automain
def archimedes_experiment(_log, coefficients):
    
    dataset = load_dataset()
    _log.info("Source data loaded from ET and CT")

    rd_df = calc_residues(dataset, coefficients)
    _log.info("Residues calculation done")

    corrMatrix = sns.heatmap(rd_df.corr("pearson"), square=True, cmap="viridis")
    _log.info("Correlation matrix calculation done")

    fig = corrMatrix.get_figure()
    fig.savefig("/tmp/corrMatrix.png")
    ex.add_artifact("/tmp/corrMatrix.png")
    _log.info("Artifacts saved")
