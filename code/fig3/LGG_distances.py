import powerlaw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, euclidean


DATAFOLDER = "../../data/"

for dataset in ['plugin', 'plugin2', 'apps2','biotech','videogames','smartphone']:
    df = pd.read_csv(DATAFOLDER + 'full_' + dataset + '.csv', encoding="latin-1")

    for i in range(len(df)):
        if df.loc[i, 'country'] == 'USA':
            df.loc[i, 'country'] = 'US'

    for country in ['US']:#, 'JP']:
        print(dataset, country)
        locations = []
        latitudes = []
        longitudes = []
        df_country = df[df['country'] == country]
        df_country.reset_index(drop=True,inplace=True)

        for i in range(len(df_country)):
            city = (df_country.loc[i,'longitude'], df_country.loc[i,'latitude'])
            if city not in locations:
                locations.append(city)
                longitudes.append(city[0])
                latitudes.append(city[1])

        print(dataset, len(locations))
        '''distances = np.zeros((len(locations),len(locations)))
        for i, a in enumerate(locations):
            for j, b in enumerate(locations):
                if i != j:
                    distances[i, j] = euclidean(a, b)
                    distances[j, i] = distances[i, j]

        np.save(dataset+'_'+country, distances)'''