# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 23:13:37 2019

@author: spolex
"""
from loaders import Properties, path, pd

schema = Properties.schema

def read(root, file, schema=schema):
    filename = path.join(root, file)
    suffix = file[-6:-4]
    if path.exists(filename):
        df = pd.read_csv(filename, sep="\s+", header=None, names=schema, skiprows=1)
        if "control" in file:
            df[Properties.subject_id] = file[0:8] + "_" + suffix
        else:
            df[Properties.subject_id] = file[0:4] + "_" + suffix if "T" in file else file[0:3] + "_" + suffix
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

def read_biodarw(filename):
    df = pd.read_csv(filename, sep="\s+", header=None, names=Properties.schema, skiprows=1)
    return df

def load_biodarw(index, abs_paths):
    dataset = None
    for i, filename in zip(index, abs_paths) :
        tmp_df = read_biodarw(filename)
        tmp_df['subject_id'] = i
        dataset = pd.concat([tmp_df,dataset])
    return dataset

    

