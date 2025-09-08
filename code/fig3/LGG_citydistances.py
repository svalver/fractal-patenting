import powerlaw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, euclidean

df = pd.read_csv('US_cities.csv', delimiter=',')
for total in [1307, 2464, 5000]:
    for replicate in range(10):
        seed = int(np.random.random()*1000)
        # Random sampling from df_main with total rows using population as weights
        probabilities = df['population'] / df['population'].sum()
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

        np.save(str(total)+'_Cities_Pop_'+str(replicate), distances)

        seed = int(np.random.random() * 1000)
        df_random = df.sample(n=total, weights=None, random_state=seed)
        df_random.reset_index(drop=True, inplace=True)

        locations = []
        for i in range(len(df_random)):
            city = (df_random.loc[i,'longitude'], df_random.loc[i,'latitude'])
            if city not in locations:
                locations.append(city)

        distances = np.zeros((len(locations),len(locations)))
        for i, a in enumerate(locations):
            for j, b in enumerate(locations):
                if i != j:
                    distances[i, j] = euclidean(a, b)
                    distances[j, i] = distances[i, j]

        np.save(str(total)+'_Cities_Random_'+str(replicate), distances)