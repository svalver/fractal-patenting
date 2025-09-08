import networkx as nx
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, euclidean


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



plt.figure(figsize=(3.5, 3))
color_vector = sns.color_palette("summer", n_colors=5)

for c_index, dataset in enumerate(['30000','10000','3000','1000']):
    df = pd.read_csv('LGG_'+dataset+'.txt', delimiter=' ')

    x = list(df['D'])
    y = list(df['N_c'])

    plt.plot(x, y, '-', color=color_vector[c_index+1], label='n='+dataset)

    if dataset == '30000':
        x = x[52:76]
        y = y[52:76]

        popt, pcov = curve_fit(func_powerlaw, x, y, maxfev=10 ** 6)
        ci_power_law = get_confidence_intervals(popt, pcov)
        error = (ci_power_law[0][0] - popt[0])
        plt.plot(x, func_powerlaw(x, *popt), color='r', label=r' $\alpha=$' + str(round(popt[0], 2)) + r'$\pm$' + str(round(abs(error), 3)))

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('US_cities_subsample.pdf')
plt.close()
