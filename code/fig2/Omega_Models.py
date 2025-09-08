import numpy as np
from collections import defaultdict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


# Initialize
data = pd.DataFrame(columns=['Dataset','N','Omega'])

Omega_V = 1
A_V = 400
K_V = 0
D_V = 0

Omega_F = 1
A_F = 300
K_F = 0
D_F = 0

for t in range(35000):
    if t % 1000 == 0:
        print(t)
    A_V += 0.5 - Omega_V
    K_V += Omega_V + 0.75
    D_V += Omega_V
    Omega_V = A_V /(A_V+K_V)


    df_ = pd.DataFrame([['Constant v', t+1, Omega_V, D_V]], columns=['Dataset', 'N', 'Omega','D'])
    data = pd.concat((data, df_))

    A_F += - Omega_F
    K_F += Omega_F + 0.25
    D_F += Omega_F
    Omega_F = A_F /(A_F+K_F)


    df_ = pd.DataFrame([['Finite expansion', t+1, Omega_F,D_F]], columns=['Dataset', 'N', 'Omega','D'])
    data = pd.concat((data, df_))


data.to_csv('Omega_models.csv')