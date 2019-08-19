# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:17:18 2019

@author: isancmen
"""
import os
from interfaces.reader_and_writer import load

filename="../output/archimedean_ds.h5"
mode='r'

def main():
    if os.path.exists(filename):
        train_rd = load(filename,'train_rd',mode)
        train_r = load(filename,'train_r',mode)
        labels = load(filename,'labels',mode)
        print(train_rd[:,-1])
        print(train_r[:,-1])
        print(labels)
        print(train_rd.shape,train_r.shape,labels.shape)
    else: print(os.path.exists(filename))
        

if __name__ == "__main__":
    main()