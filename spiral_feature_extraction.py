# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:31:33 2019

@author: isancmen
"""

from interfaces.arq_loader import load_arquimedes_dataset
from interfaces.reader_and_writer import save
from feature_extract.arq_features import *
import numpy as np
from datetime import datetime

##Logging configuration
import logging
import logging.config

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

def extract_radio(L):
    return radio(L['x'],L['y'])

def extract_residuos(L):
    x =  residuos(L['x'])
    y =  residuos(L['y'])
    return radio(x,y)

def extract_features_of(L,ts=None):
    return [
    samp_ent(L)
    ,mean_abs_val(L)
    ,L.var()
    ,root_mean_square(L)
    ,log_detector(L)
    ,wl(L)
    ,L.std()
    ,diff_abs_std(L)
    ,higuchi(L)
    ,mfl(L)
    ,myo(L,ts)
    ,iemg(L)
#    ,ssi(L)
#    ,zc(L)
#    ,ssc(L)
#    ,wamp(L)
#    ,p_max(L)
#    ,f_max(L)
#    ,mp(L)
#    ,tp(L)
#    ,meanfreq(L)
#    ,medfreq(L)
#    ,std_psd(L)
#    ,mmnt(L,order=1)
#    ,mmnt(L,order=2)
#    ,mmnt(L,order=3)
#    ,kurt(L)
#    ,skw(L)
#    ,autocorr(L)
    ]

def main():
    logger.debug("Starting feature extraction from archimedean spirals")
    logger.debug("Loading control files")
    ct = load_arquimedes_dataset(filenames_file,root_ct)
    logger.debug("Loading ET files %d",len(ct))
#    et = load_arquimedes_dataset(filenames_file,root_et)
    ct_ts= list(map(lambda df: df['timestamp'],ct))
#    et_ts= list(map(lambda df: df['timestamp'],et))
#    r_ct = list(map(extract_radio,ct))
#    r_ct_fe = list(map(extract_features_of,r_ct,ct_ts))
#    hf.create_dataset('r_ct_fe', data=r_ct_fe)
    rd_ct = list(map(extract_residuos,ct))
    logger.debug("CT's residual radio calculation %d",len(rd_ct))
    rd_ct_fe = np.array(list(map(extract_features_of,rd_ct,ct_ts)))
    logger.debug("CT's feature extraction %i",len(rd_ct_fe[0]))
    logger.debug("Saving CT's residual feature extraction in "+h5file)
    save(h5file,'rd_ct_fe',rd_ct_fe)
#    hf.create_dataset('rd_ct_fe', data=rd_ct_fe)
#    r_et = list(map(extract_radio,et))
#    r_et_fe = list(map(extract_features_of,r_et,et_ts))
#    hf.create_dataset('r_et_fe', data=r_et_fe)
#    rd_et = list(map(extract_residuos,et))
#    rd_et_fe = list(map(extract_features_of,rd_et,et_ts))
#    hf.create_dataset('rd_et_fe', data=rd_et_fe)

if __name__ == "__main__":
    main()