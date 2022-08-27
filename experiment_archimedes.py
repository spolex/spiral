# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:58:30 2019

@author: isancmen
"""
import swifter
import time
import argparse
import logging.config
from logging.handlers import RotatingFileHandler
from properties import properties as Properties
import pandas as pd
from pandas import HDFStore
from loaders.biodarw import load_arquimedes_dataset

from preprocess.biodarw_feature_extraction import extract_radio, extract_residues, extract_features_of
from scipy.signal import resample
from skrebate import ReliefF
from sklearn.feature_selection import RFE

from analysis import svm_cv, clf_ho, clf_loo, svm_deap

logger = logging.getLogger('MainLogger')
logging.config.fileConfig(Properties.log_conf_path)
formatter = logging.Formatter("%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
                              , '%Y-%m-%d %H:%M:%S')
logFile = Properties.log_file_path
logger.info("Log file path location " + logFile)
rh = RotatingFileHandler(logFile, mode='a', maxBytes=2.4 * 1024 * 1024,
                         backupCount=5, encoding="UTF-8", delay=0)
rh.setFormatter(formatter)
logger.addHandler(rh)


def main():
    args = parser.parse_args()

    start_time = time.time()

    for coefficient in Properties.coefficients:
        # load data
        h5filename = Properties.h5file + str(coefficient) + Properties.extension
        logger.info("File relative path " + h5filename)
        hdf = HDFStore(h5filename)

        if args.load:
            logger.info("Loading Controls files")
            ct = load_arquimedes_dataset(Properties.file_list_path, Properties.ct_root_path)
            ct[Properties.labels] = 'ct'
            logger.info("Loading ET files")
            et = load_arquimedes_dataset(Properties.file_list_path, Properties.et_root_path)
            et[Properties.labels] = 'et'
            dataset = pd.concat([ct, et])
            hdf.put('source/dataset', dataset, data_columns=True)
            labels = dataset[['subject_id', 'labels']].drop_duplicates().set_index('subject_id')['labels']
            y = labels.reset_index()['labels']
            hdf.put('source/labels', labels, data_columns=True)
        else:
            logger.info("Reading dataset from %s", h5filename)
            dataset = hdf.get('source/dataset')
            labels = hdf.get('source/labels')
            y = labels.reset_index()['labels']

        # transform data
        if args.transform:
            if args.radius:
                r = dataset.groupby(Properties.subject_id).apply(extract_radio)\
                    .swifter.apply(resample, num=Properties.resample)
                r_df = pd.DataFrame.from_records(zip(r.index, r.values))
                print(r.head())
                hdf.put('results/radius/r', r_df, data_columns=True)

            if args.residues:
                rd = dataset.groupby(Properties.subject_id).apply(extract_residues, c=coefficient)\
                    .swifter.apply(resample, num=Properties.resample)
                rd_df = pd.DataFrame.from_records(zip(rd.index, rd.values))
                hdf.put('results/residues/rd', rd_df, data_columns=True)

        # feature engineering
        if args.preprocess:
            if args.radius:
                r = r if args.transform else hdf.get('results/radius/r').T
                r_fe = r.swifter.apply(extract_features_of, axis='columns')
                r_fe_df = pd.DataFrame.from_records(zip(r_fe.index, r_fe.values)).T
                r_fe_df.columns = Properties.features_names
                hdf.put('results/radius/features', r_fe_df, data_columns=True)
                if args.relief:
                    # feature selection
                    fltr = RFE(ReliefF(), n_features_to_select=5, step=1)
                    fltr_r_fe_df = fltr.fit_transform(r_fe_df, y)
                    hdf.put('results/radius/relief_features', fltr_r_fe_df, data_columns=True)

            if args.residues:
                rd = rd if args.transform else hdf.get('results/residues/rd')
                rd_fe = rd.swifter.apply(extract_features_of, axis='columns')
                rd_fe_df = pd.DataFrame.from_records(zip(rd_fe.index, rd_fe.values)).T
                rd_fe_df.columns = Properties.features_names
                hdf.put('results/residues/features', rd_fe_df, data_columns=True)
                if args.relief:
                    # feature selection
                    fltr = RFE(ReliefF(), n_features_to_select=5, step=1)
                    fltr_rd_fe_df = fltr.fit_transform(rd_fe_df, y)
                    hdf.put('results/residues/relief_features', fltr_rd_fe_df, data_columns=True)

        # analysis
        if args.analysis:
            if args.relief:
                X = hdf.get('results/residues/relief_features') if args.residues else hdf.get('results/radius/features')
            else:
                X = hdf.get('results/residues/features') if args.residues else hdf.get('results/radius/features')
            svm_cv.svm_cv(X, y) if args.svm_cv else True
            svm_deap.svm_ga(X, y) if args.svm_deap else True
            clf_loo.analysis_loo(X, y) if args.alo else True
            clf_ho.analysis_ho(X, y) if args.aho else True

    hdf.close()
    elapsed_time = time.time() - start_time
    logger.info("Total elapsed time is %s", elapsed_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Feature engineering toolbox for "
                                                 "Archimedean's Spiral images processing")

    parser.add_argument("-r", "--radius", help='If present radius of the signals', action='store_true')
    parser.add_argument("-u", "--residues", help='If present residues of the signals', action='store_true')

    parser.add_argument("-l", "--load", help='If present dataset is loaded from source origin and saved into hdf5 '
                        , action='store_true')

    parser.add_argument("-t", "--transform", help='If present radius or residues feature is being executed '
                        , action='store_true')

    parser.add_argument("-p", "--preprocess", help='If present data preprocess is being executed ', action='store_true')

    parser.add_argument("-f", "--relief", help='If present features selection is being executed ', action='store_true')

    parser.add_argument("-i", "--analysis", help='If present data analysis is being executed', action='store_true')

    parser.add_argument("-v", "--svm_cv", help='If present SVM CV grid search is being executed evaluation strategy',
                        action='store_true')
    parser.add_argument("-d", "--svm_deap", help='If present SVM GA search is being executed evaluation strategy',
                        action='store_true')

    parser.add_argument("-o", "--aho", help='If present analysis is being executed with holdout evaluation strategy',
                        action='store_true')
    parser.add_argument("-z", "--alo", help='If present analysis is being executed with CV Leaving One Out '
                                            'evaluation strategy', action='store_true')

    main()