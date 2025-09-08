import networkx as nx
import powerlaw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, euclidean

d_list = [0.01]
while d_list[-1] < 1.5:
    d_list.append(d_list[-1]*1.1)

country = 'US'

for dataset in ['videogames','biotech']:
    dataframe = pd.DataFrame(columns=['D', 'N_c', 'dataset', 'method'])
    #plt.figure()
    if dataset == 'biotech':
        total = 2464
    else:
        total = 1307

    Nd = []
    distances = np.load(dataset+'_'+country+'.npy')
    for d in d_list:
        print(dataset, 'data', d)
        adjacency = np.where(distances < d, 1, 0)
        G = nx.from_numpy_array(adjacency, create_using=nx.Graph)
        Nclusters = len(list(nx.connected_components(G)))/len(G.nodes)
        Nd.append(Nclusters)
        data_ = pd.DataFrame([[d, Nclusters, dataset, 'data']], columns=['D', 'N_c', 'dataset', 'method'])
        dataframe = pd.concat((dataframe, data_))
    #plt.plot(d_list, Nd, label=dataset)

    for replicate in range(5):
        Nd = []
        distances = np.load(str(total)+'_Cities_Random_'+str(replicate) + '.npy')
        for d in d_list:
            print(dataset, replicate, d)
            adjacency = np.where(distances < d, 1, 0)
            G = nx.from_numpy_array(adjacency, create_using=nx.Graph)
            Nclusters = len(list(nx.connected_components(G))) / len(G.nodes)
            Nd.append(Nclusters)
            data_ = pd.DataFrame([[d, Nclusters, dataset, 'Random sample']], columns=['D', 'N_c', 'dataset', 'method'])
            dataframe = pd.concat((dataframe, data_))
        #plt.plot(d_list, Nd, label='Random')

        Nd = []
        distances = np.load(str(total)+'_Cities_PreferentialPop_'+str(replicate) + '.npy')
        for d in d_list:
            print(dataset, replicate, d)
            adjacency = np.where(distances < d, 1, 0)
            G = nx.from_numpy_array(adjacency, create_using=nx.Graph)
            Nclusters = len(list(nx.connected_components(G))) / len(G.nodes)
            Nd.append(Nclusters)
            data_ = pd.DataFrame([[d, Nclusters, dataset, 'Population bias']], columns=['D', 'N_c', 'dataset', 'method'])
            dataframe = pd.concat((dataframe, data_))
        #plt.plot(d_list, Nd, label='Population')

    #plt.legend()
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.title(dataset)
    #plt.savefig('LGG_RPD_'+dataset+'.pdf')
    #plt.close()

    dataframe.to_csv('LGG_'+dataset+'_RPD2.csv')