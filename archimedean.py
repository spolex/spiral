# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:58:30 2019

@author: isancmen
"""

import logging
import logging.config
from datetime import datetime
import time

from preprocessing.spiral_feature_extraction import extract_features
from preprocessing.datasets_preparation import dataset_prep

#TODO extract properties to properties file
logger = logging.getLogger('MainLogger')
logging.config.fileConfig('conf/logging.conf')

fh = logging.FileHandler('log/archimedean-{:%Y-%m-%d}.log'.format(datetime.now()))
formatter = logging.Formatter('%(asctime)s ; %(levelname)-8s ; %(name)s ; %(module)s ; %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

filenames_file="E:/04-DATASOURCES/01-PHD/00-NEW/02-WEE/ETHW/ETONA.txt"
root_ct="E:/04-DATASOURCES/01-PHD/00-NEW/02-WEE/ETHW/Controles30jun14/"
root_et="E:/04-DATASOURCES/01-PHD/00-NEW/02-WEE/ETHW/Protocolo temblor"
h5file="output/archimedean.h5"

def main():
    start_time = time.time()
    extract_features(filenames_file,root_ct,root_et,h5file)
    elapsed_time = time.time() - start_time
    dataset_prep()
    logger.info("Elapsed time extracting features %s",elapsed_time)
    
if __name__ == "__main__":
    main()