# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:01:44 2019

@author: isancmen
"""
import h5py
import gc

for obj in gc.get_objects():   # Browse through ALL objects
    if isinstance(obj, h5py.File):   # Just HDF5 files
        try:
            obj.close()
        except:
            pass # Was already closed