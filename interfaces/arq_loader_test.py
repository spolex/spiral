# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 23:22:31 2019

@author: isancmen
"""
from arq_loader import load_arquimedes_dataset

#parameters
filenames_file="E:/04-DATASOURCES/01-PHD/00-NEW/02-WEE/ETHW/ETONA.txt"
root_ct="E:/04-DATASOURCES/01-PHD/00-NEW/02-WEE/ETHW/Controles30jun14/"
root_et="E:/04-DATASOURCES/01-PHD/00-NEW/02-WEE/ETHW/Protocolo temblor/"

def main():
    df = load_arquimedes_dataset(filenames_file,root_ct)
    print(df[0].head())
    

if __name__ == "__main__":
    main()