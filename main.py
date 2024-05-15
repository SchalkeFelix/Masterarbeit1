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

    x, verkehrsdaten, y, z = read_lkw_data(csv_dateipfad)
    lkw_werte = verkehrsdaten['y_continuous']

    tage_im_jahr = days_in_year(jahr)

    nummer_timestep = [x for x in range(0, int((tage_im_jahr * 24 * 60) / timedelta), 1)]
    lkw_werte = pd.DataFrame(lkw_werte, index=nummer_timestep, columns=['LKW_in_timestep'])
    alle_wochen = gesamt_df_splitten(lkw_werte, jahr)

    minutenwerte = list(range(0, 7*24*60, timedelta))

    output_folder = 'Wochen zu Clusterung'

    # Erstelle den Ordner, falls er nicht existiert
    os.makedirs(output_folder, exist_ok=True)

    for i, part_df in enumerate(alle_wochen):
        part_df.set_index(pd.Index(minutenwerte), inplace=True)
        csv_filename = os.path.join(output_folder, f'woche_{i+1}.csv')
        part_df.to_csv(csv_filename, index=False)
        print(f"Woche {i + 1}:")
        print(part_df)
        print()

    plot_verkehr(verkehrsdaten, 'Woche1')

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
