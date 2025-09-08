import networkx as nx
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def func_powerlaw(x, m, c):
    return c*x**m


def get_confidence_intervals(popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    n_params = len(popt)
    z = 1.96  # 95% confidence interval (alpha=0.05)
    conf_intervals = []
    for i in range(n_params):
        conf_intervals.append((popt[i] - z * perr[i], popt[i] + z * perr[i]))
    return conf_intervals


country = 'US'

for dataset in ['videogames','biotech']:
    df = pd.read_csv('LGG_'+dataset+'_RPD.csv')
    plt.figure(figsize=(3.5,3))
    sns.lineplot(x='D', y='N_c', hue='method',style='method', ci='sd',
                 hue_order=['No bias', 'Population bias', 'data'], palette='flare', data=df)

    df_ = df[df['method'] == 'data']
    print(df.head())
    x = list(df_['D'])[93:224]
    y = list(df_['N_c'])[93:224]
    print(len(x), len(y))
    popt, pcov = curve_fit(func_powerlaw, x, y, maxfev=10 ** 6)
    ci_power_law = get_confidence_intervals(popt, pcov)
    error = (ci_power_law[0][0] - popt[0])
    plt.plot(x, func_powerlaw(x, *popt), color='r', label=r' $\alpha=$' + str(round(popt[0], 2)) + r'$\pm$' + str(round(abs(error), 3)))


    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    #plt.title(dataset)
    plt.tight_layout()
    plt.savefig('RPD2_'+dataset+'.pdf')
    plt.close()
