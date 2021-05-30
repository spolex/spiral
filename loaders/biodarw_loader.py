# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 23:13:37 2019

@author: spolex
"""
from loaders import Ingredient, Properties, path, pd

data_ingredient = Ingredient('loader')
data_ingredient.add_config("conf/sacred_config.json")

schema = eval(Properties['schema'])

def read(root, file, schema=schema):
    filename = path.join(root, file)
    suffix = file[-6:-4]
    if path.exists(filename):
        df = pd.read_csv(filename, sep="\s+", header=None, names=schema, skiprows=1)
        if "control" in file:
            df[Properties["subject_id"]] = file[0:8] + "_" + suffix
        else:
            df[Properties["subject_id"]] = file[0:4] + "_" + suffix if "T" in file else file[0:3] + "_" + suffix
        return df


def read_filenames_from(file):
    filenames = []
    with open(file) as f:
        for line in f:
            filenames.append(line.replace("\n", ""))
    return filenames


def load_arquimedes_dataset(filenames_file, root):
    files = read_filenames_from(filenames_file)
    df = pd.concat(list(filter(lambda x: x is not None, list(map(lambda file: read(root, file), files)))))
    return df

@data_ingredient.config
def archimedes_config(data_dir, coefficients):
    h5filename = path.join(data_dir,"archimedean-"+str(coefficients)+".h5")

@data_ingredient.capture
def get_h5_filename(data_dir, coefficients):
    return path.join(data_dir,"archimedean-"+str(coefficients)+".h5")

@data_ingredient.capture
def load_dataset(file_list_path, ct_root_path, et_root_path, labels, _log):
    _log.info("Loading Controls subjects handwriting raw files")
    ct = load_arquimedes_dataset(file_list_path, ct_root_path)
    ct[labels] = 'ct'
    _log.debug(ct.head())
    _log.info("Loading ET subjects handwriting raw files")
    et = load_arquimedes_dataset(file_list_path, et_root_path)
    et[labels] = 'et'
    return pd.concat([ct, et])


