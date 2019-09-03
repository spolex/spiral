# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:58:30 2019

@author: isancmen
"""

import logging.config
from logging.handlers import RotatingFileHandler
from datetime import datetime
import time

from preprocessing.spiral_feature_extraction import extract_features,extract_rr

# TODO extract properties to properties file
logger = logging.getLogger('MainLogger')
logging.config.fileConfig('conf/logging.conf')

formatter = logging.Formatter("%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
                              , '%Y-%m-%d %H:%M:%S')

logFile = 'log/archimedean-{:%Y-%m-%d}.log'.format(datetime.now())

rh = RotatingFileHandler(logFile, mode='a', maxBytes=2.4 * 1024 * 1024,
                         backupCount=5, encoding="UTF-8", delay=0)

rh.setFormatter(formatter)
logger.addHandler(rh)

filenames_file = "E:/04-DATASOURCES/01-PHD/00-NEW/02-WEE/ETHW/ETONA.txt"
root_ct = "E:/04-DATASOURCES/01-PHD/00-NEW/02-WEE/ETHW/Controles30jun14/"
root_et = "E:/04-DATASOURCES/01-PHD/00-NEW/02-WEE/ETHW/Protocolo temblor"
h5file = "output/archimedean.h5"


def main():
    start_time = time.time()
    extract_rr(filenames_file, root_ct, root_et, h5file, samples=4096)
#    extract_features(h5file)
    elapsed_time = time.time() - start_time
    logger.info("Elapsed time extracting features %s", elapsed_time)


if __name__ == "__main__":
    main()
