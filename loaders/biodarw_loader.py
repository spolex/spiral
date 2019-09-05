# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 23:13:37 2019

@author: isancmen
"""
import pandas as pd
import os

schema = ['x', 'y', 'timestamp', 'pen_up', 'azimuth', 'altitude', 'pressure']


def read(filename, schema=schema):
    if os.path.exists(filename):
        return pd.read_csv(filename, sep="\s+", header=None, names=schema, skiprows=1)


def read_filenames_from(file):
    filenames = []
    with open(file) as f:
        for line in f:
            filenames.append(line.replace("\n", ""))
    return filenames


def load_arquimedes_dataset(filenames_file, root):
    files = read_filenames_from(filenames_file)
    return list(filter(lambda x: x is not None, list(map(lambda file: read(root + file), files))))
