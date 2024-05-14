# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gurobipy as gp
import datetime

import matplotlib.pyplot as plt
import pandas as pd

from approximation import *
from plots import *

erster_montag = 210104 #in Form 'YYMMDD'
jahr = 2021

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    gesamte_anzahl_pro_woche, max_pro_woche, wochenliste = wochenweise_iterieren(erster_montag, jahr)

    data = {'max. Anzahl in Woche': max_pro_woche, 'Gesamtanzahl pro Woche': gesamte_anzahl_pro_woche}
    df = pd.DataFrame(data, index=wochenliste)

    plt.scatter(df['max. Anzahl in Woche'], df['Gesamtanzahl pro Woche'], color='blue', label='Datenpunkte')
    #plt.show()

    print(kmeans_clustering(df, 3, 100))

