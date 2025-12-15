# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 09:23:37 2025

@author: michel
"""

import h5py
import numpy as np

with h5py.File(r"D:\Michel\DAS-dataset\data\car\auto_2023-04-17T124152+0100.h5", "r") as f:
    grp = f["Acquisition"]
    print("Sous-éléments dans 'Acquisition' :", list(grp.keys()))

   
    data = grp["Raw[0]"]['RawData']  
    custom = grp["Raw[0]"]['Custom']
    print("Shape du dataset :", data.shape)
    print("Type du dataset :", type(data))
    print("Exemple de valeurs :", data[:5])
    # print('Custom : ', type(custom) )



#bitmap


import numpy as np

# Charger le fichier .npy
data = np.load(r"D:\Michel\DAS-dataset\data\car\auto_2023-04-17T124152+0100.npy")

print("Type :", type(data))
print("Shape :", data.shape)
print("Exemple de valeurs :", data[:10])



