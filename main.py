# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gurobipy as gp
import datetime

import matplotlib.pyplot as plt
import pandas as pd

from methods import *
from plots import *
from Initialiserung import *



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    gesamte_anzahl_pro_woche, max_pro_woche, wochenliste = wochenweise_iterieren(erster_montag, jahr)

    data = {'max. Anzahl in Woche': max_pro_woche, 'Gesamtanzahl pro Woche': gesamte_anzahl_pro_woche}
    df = pd.DataFrame(data, index=wochenliste)

    plt.scatter(df['max. Anzahl in Woche'], df['Gesamtanzahl pro Woche'], color='blue', label='Datenpunkte')
    #plt.show()

    cluster_array, egal = kmeans_clustering(df, anzahl_cluster, 1000)
    print(cluster_array)

    print(cluster_zuordnen(cluster_array))
    """
    x, verkehrsdaten, y, z = read_lkw_data(csv_dateipfad)
    lkw_werte = verkehrsdaten['y_continuous']

    minutenwerte = [x for x in range(0, 525596, 5)]
    print(minutenwerte)
    print(len(minutenwerte))
    print(len(lkw_werte))
    print(lkw_werte)
    lkw_werte = pd.DataFrame(lkw_werte, index=minutenwerte, columns=['LKW_in_timestep'])
    dummy = gesamt_df_splitten(lkw_werte, jahr)

    plot_verkehr(verkehrsdaten, 'Woche1')