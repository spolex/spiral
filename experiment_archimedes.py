# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:58:30 2019

@author: isancmen
"""
import time
import argparse
import logging.config
from logging.handlers import RotatingFileHandler
from datetime import datetime
from loaders.reader_and_writer import load
from biodarw_feature_extraction import extract_features, extract_rr
from preprocess.biodarw2trainingdataset import dataset_prep
from analysis import svm_cv, clf_ho, clf_loo, svm_deap
import sys

# TODO extraer a properties para poder automatizar experimentos

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

mode = 'r'
train_dataname = 'train_rd'

# param_grid = {"svc__kernel": ["rbf"],
#              "svc__C": np.logspace(-5, 5, num=25, base=10),
#              "svc__gamma": np.logspace(-9, 9, num=25, base=10)}

coefficients = range(17, 18, 1)
h5file = "output/archimedean-"
extension = ".h5"
filename_ds = "output/archimedean_ds-"


def main():
    args = parser.parse_args()

    start_time = time.time()

    if args.radius:
        for coefficient in coefficients:
            h5filename = h5file + str(coefficient) + extension
            logger.debug("File relative path " + h5filename)
            extract_rr(filenames_file, root_ct, root_et, h5filename, coeff=coefficient, samples=4096)

    if args.extract:
        for coefficient in coefficients:
            h5filename = h5file + str(coefficient) + extension
            logger.debug("File relative path " + h5filename)
            extract_features(h5filename)

    if args.prep:
        for coefficient in coefficients:
            h5filename = h5file + str(coefficient) + extension
            h5filename_ds = filename_ds + str(coefficient) + extension
            logger.debug("File relative path " + h5filename)
            logger.debug("File DS relative path " + h5filename_ds)
            dataset_prep(h5filename, h5filename_ds)

    if args.svm_cv or args.alo or args.aho or args.svm_deap:
        for coefficient in coefficients:
            h5filename_ds = filename_ds + str(coefficient) + extension
            logger.debug("File DS relative path " + h5filename_ds)
            X = load(h5filename_ds, train_dataname, mode)
            y = load(h5filename_ds, 'labels', mode)
            clf_ho.analysis_ho(X, y) if args.aho else True
            clf_loo.analysis_loo(X, y) if args.alo else True
            svm_cv.svm_cv(X, y, rfe=True) if args.svm_cv else True
            svm_deap.svm_ga(X, y, rfe=True) if args.svm_deap else True

    elapsed_time = time.time() - start_time
    logger.info("Total elapsed time is %s", elapsed_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature engineering toolbox for Archimedean's Spiral images processing")
    parser.add_argument("-r", "--radius", help='If present radius and residual radious of the signals '
                                               'is being calculated ', action='store_true')
    parser.add_argument("-e", "--extract", help='If present feature extraction is being executed ', action='store_true')
    parser.add_argument("-p", "--prep", help='If present data preparation is being executed ', action='store_true')
    parser.add_argument("-o", "--aho", help='If present analysis is being executed with holdout evaluation strategy',
                        action='store_true')
    parser.add_argument("-l", "--alo", help='If present analysis is being executed with CV Leaving One Out '
                                            'evaluation strategy', action='store_true')
    parser.add_argument("-v", "--svm_cv", help='If present SVM CV grid search is being executed evaluation strategy',
                        action='store_true')
    parser.add_argument("-d", "--svm_deap", help='If present SVM GA search is being executed evaluation strategy',
                        action='store_true')

    orig_stdout = sys.stdout
    f = open('output/experiment.log', 'w')
    sys.stdout = f
    main()
    sys.stdout = orig_stdout
    f.close()
