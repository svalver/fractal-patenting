import powerlaw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, euclidean


df = pd.read_csv('US_cities.csv', delimiter=',')
#Preferential attachment
alpha = 100
p = np.asarray([x**alpha for x in list(df['population'])])
probabilities = p / np.sum(p)

for total in [1307, 2464]:
    for replicate in range(5):
        print(total, replicate)
        seed = int(np.random.random()*1000)
        # Random sampling from df_main with total rows using population as weights
        df_population = df.sample(n=total, weights=probabilities, random_state=seed)
        df_population.reset_index(drop=True,inplace=True)

        locations = []
        for i in range(len(df_population)):
            city = (df_population.loc[i,'longitude'], df_population.loc[i,'latitude'])
            if city not in locations:
                locations.append(city)

        distances = np.zeros((len(locations),len(locations)))
        for i, a in enumerate(locations):
            for j, b in enumerate(locations):
                if i != j:
                    distances[i, j] = euclidean(a, b)
                    distances[j, i] = distances[i, j]

        np.save(str(total)+'_Cities_PreferentialPop_'+str(replicate), distances)
