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

    ### CLUSTERING ###

    # Verkehrsdaten aufrufen
    x, verkehrsdaten, y, z = read_lkw_data(csv_dateipfad)
    lkw_werte = verkehrsdaten['y_continuous']

    # Jahresverlauf in einzelne Wochen splitten
    tage_im_jahr = days_in_year(jahr)
    nummer_timestep = [x for x in range(0, int((tage_im_jahr * 24 * 60) / timedelta), 1)]
    lkw_werte = pd.DataFrame(lkw_werte, index=nummer_timestep, columns=['LKW_in_timestep'])
    alle_wochen = gesamt_df_splitten(lkw_werte, jahr)

    # Dataframes zwischenspeichern
    output_folder = 'Wochen zu Clusterung'
    os.makedirs(output_folder, exist_ok=True)

    # Gesamtübersicht erstellen
    anzahl_wochen = count_full_weeks(jahr)
    gesamt_uebersicht = pd.DataFrame(columns=["max. Wert", "Gesamtmenge"])

    # Gesamtübersicht befüllen
    for i, part_df in enumerate(alle_wochen):
        #part_df.set_index(pd.Index(minutenwerte), inplace=True)
        csv_filename = os.path.join(output_folder, f'woche_{i+1}.csv')
        part_df.to_csv(csv_filename, index=False)
        max_wert = float(part_df['LKW_in_timestep'].max())
        gesamtmenge = float(part_df['LKW_in_timestep'].sum())
        gesamt_uebersicht.loc[f"Woche {i+1}"] = max_wert, gesamtmenge

        print(f"Woche {i + 1}:")
        print(part_df)
        print()

    # K-Means-Clustering durchführen und Cluster plotten
    plt.scatter(gesamt_uebersicht['max. Wert'], gesamt_uebersicht['Gesamtmenge'], color='blue', label='Datenpunkte')
    cluster_array, egal = kmeans_clustering(gesamt_uebersicht, anzahl_cluster, 1000)
    print(cluster_zuordnen(cluster_array))
    geclusterte_wochen = cluster_zuordnen(cluster_array)

    # Bespielwochen erstellen
    DUMMY = 0
    beispielwochen_berechnen(geclusterte_wochen)
    ### CLUSTERING ENDE ###



