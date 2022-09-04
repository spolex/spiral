# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:58:30 2019

@author: isancmen
"""
from os import path
import swifter
import time
import argparse
import logging.config
from logging.handlers import RotatingFileHandler
from properties import properties as Properties
import pandas as pd
from pandas import HDFStore
from loaders.biodarw import load_biodarw
from preprocess.biodarw_feature_extraction import extract_radio, extract_residues, extract_features_of
from scipy.signal import resample
from skrebate import ReliefF
from sklearn.feature_selection import RFE
import numpy as np

from analysis import svm_cv, clf_ho, clf_loo, svm_deap

from datetime import date

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
    today=date.today().strftime("%Y%m%d")


    for coefficient in Properties.coefficients:
        filename = Properties.file + str(today) + Properties.extension

        logger.debug("Loading metadata...")
        metadf=pd.read_csv(Properties.metadata_path,index_col=0)
        logger.debug(metadf.level.head(10))

        if args.load:
            logger.info("Loading data files")
            dataset=load_biodarw(metadf.index, metadf['abs_path'])
            logger.debug(dataset.head(10))
            logger.debug(metadf.temblor)
            dataset.to_csv(filename)
            metadf.level.to_csv(Properties.level_filename.format(today))
            metadf.temblor.to_csv(Properties.label_filename.format(today))
        elif path.exists(filename):
            logger.info("Reading dataset from %s", filename)
            dataset = pd.read_csv(filename,index_col=0)
        logger.debug("Loading labels")
        y = metadf.level

        # transform data
        if args.transform:
            if args.radius:
                r = dataset.groupby('subject_id').apply(extract_radio)\
                    .swifter.apply(resample, num=Properties.resample)
                r_df = pd.DataFrame(r.tolist(), index=r.index)
                r_df.to_csv(Properties.r_filename.format(today))

            if args.residues:
                rd = dataset.groupby('subject_id').apply(extract_residues, c= coefficient)\
                    .swifter.apply(resample, num=Properties.resample)
                rd_df = pd.DataFrame(rd.tolist(), index=rd.index)
                print(rd_df.head())
                rd_df.to_csv(Properties.rd_filename.format(coefficient , today))

        # feature engineering
        if args.preprocess:
            if args.radius:
                r = r_df if args.transform else pd.read_csv(Properties.r_filename.format(today), index_col=0)
                r_fe = r.swifter.apply(extract_features_of, axis='columns')
                r_fe_df = pd.DataFrame(r_fe.to_list(), index=r_fe.index)
                r_fe_df.columns = Properties.features_names
                r_fe_df.to_csv(Properties.r_feat_filename.format(today))
                if args.relief:
                    # feature selection
                    selector = RFE(ReliefF(), n_features_to_select=5, step=1)
                    selector_r = selector.fit(r_fe_df, y)
                    idx_r = np.where(selector_r.ranking_ == 1)
                    r_fe_df_relief_cols = r_fe_df.columns[idx_r]
                    r_fe_df[r_fe_df_relief_cols].to_csv(Properties.r_feat_relief_filename.format(today))

            if args.residues:
                rd = rd_df if args.transform else pd.read_csv(Properties.rd_filename.format(coefficient, today), index_col=0)
                rd_fe = rd.swifter.apply(extract_features_of, axis='columns')
                rd_fe_df = pd.DataFrame(rd_fe.to_list(), rd_fe.index)
                rd_fe_df.columns = Properties.features_names
                rd_fe_df.to_csv(Properties.rd_feat_filename.format(coefficient, today))
                if args.relief:
                    # feature selection
                    selector = RFE(ReliefF(), n_features_to_select=5, step=1)
                    selector_rd = selector.fit(rd_fe_df, y)
                    idx_rd = np.where(selector_rd.ranking_ == 1)
                    rd_fe_df_relief_cols = rd_fe_df.columns[idx_rd]
                    rd_fe_df[rd_fe_df_relief_cols].to_csv(Properties.rd_feat_relief_filename.format(coefficient,today))

        # analysis
        if args.analysis:
            if args.relief:
                X = pd.read_csv(Properties.rd_feat_relief_filename.format(coefficient, today), index_col=0) if args.residues else pd.read_csv(Properties.r_feat_relief_filename.format(today), index_col=0)
            else:
                X = pd.read_csv(Properties.rd_feat_filename.format(coefficient, today), index_col=0) if args.residues else pd.read_csv(Properties.r_feat_filename.format(today), index_col=0)
            svm_cv.svm_cv(X, y) if args.svm_cv else True
            svm_deap.svm_ga(X, y) if args.svm_deap else True
            clf_loo.analysis_loo(X, y) if args.alo else True
            clf_ho.analysis_ho(X, y) if args.aho else True

    elapsed_time = time.time() - start_time
    logger.info("Total elapsed time is %s", elapsed_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Feature engineering toolbox for "
                                                 "Archimedean's Spiral images processing")

    parser.add_argument("-r", "--radius", help='If present radius of the signals', action='store_true')
    parser.add_argument("-u", "--residues", help='If present residues of the signals', action='store_true')

    parser.add_argument("-l", "--load", help='If present dataset is loaded from source origin and saved into csv '
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