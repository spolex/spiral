# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:17:18 2019

@author: isancmen
"""
import os
from interfaces.reader_and_writer import load

filename="../output/archimedean.h5"

dataset='rd_ct_fe'
mode='r'

def main():
    if os.path.exists(filename):
        df = load(filename,dataset,mode)
        print(df.shape)
        print(df[:,-4])
    else: print(os.path.exists(filename))
        

if __name__ == "__main__":
    main()