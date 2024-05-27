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

    if wochen_clustern:
        # Verkehrsdaten aufrufen
        x, verkehrsdaten, y, z = read_lkw_data(csv_dateipfad)
        lkw_werte = verkehrsdaten['y_continuous']

        # Jahresverlauf in einzelne Wochen splitten
        tage_im_jahr = days_in_year(jahr)
        nummer_timestep = [x for x in range(0, int((tage_im_jahr * 24 * 60) / timedelta), 1)]
        lkw_werte = pd.DataFrame(lkw_werte, index=nummer_timestep, columns=['LKW_in_timestep'])
        alle_wochen = gesamt_df_in_wochen_splitten(lkw_werte, jahr)

        # Dataframes zwischenspeichern und Gesamtübersicht befüllen
        output_folder = 'Wochen zu Clusterung'
        os.makedirs(output_folder, exist_ok=True)

        anzahl_wochen = count_full_weeks(jahr)
        gesamt_uebersicht = pd.DataFrame(columns=["max. Wert", "Gesamtmenge"])

        for i, part_df in enumerate(alle_wochen):
            csv_filename = os.path.join(output_folder, f'woche_{i+1}.csv')
            part_df.to_csv(csv_filename, index=False)
            max_wert = float(part_df['LKW_in_timestep'].max())
            gesamtmenge = float(part_df['LKW_in_timestep'].sum())
            gesamt_uebersicht.loc[f"Woche {i+1}"] = max_wert, gesamtmenge

        # K-Means-Clustering durchführen und Cluster plotten
        plt.scatter(gesamt_uebersicht['max. Wert'], gesamt_uebersicht['Gesamtmenge'], color='blue', label='Datenpunkte')
        cluster_array, egal = kmeans_clustering(gesamt_uebersicht, anzahl_cluster_wochen, 1000)
        geclusterte_wochen = cluster_zuordnen(cluster_array)

        # Bespielwochen erstellen

        beispielwochen = beispielwochen_berechnen(geclusterte_wochen)
        beispielwochen_plotten(beispielwochen, anzahl_cluster_wochen)


    elif tage_clustern:
        # Verkehrsdaten aufrufen
        x, verkehrsdaten, y, z = read_lkw_data(csv_dateipfad)
        lkw_werte = verkehrsdaten['y_continuous']

        # Jahresverlauf in einzelne Tage splitten
        tage_im_jahr = days_in_year(jahr)
        nummer_timestep = [x for x in range(0, int((tage_im_jahr * 24 * 60) / timedelta), 1)]
        lkw_werte = pd.DataFrame(lkw_werte, index=nummer_timestep, columns=['LKW_in_timestep'])
        alle_tage = gesamt_df_in_tage_splitten(lkw_werte, jahr)

        #Dataframes speichern und daten speichern
        # Dataframes zwischenspeichern
        output_folder = 'Tage zu Clusterung'
        os.makedirs(output_folder, exist_ok=True)

        gesamt_uebersicht_tage = pd.DataFrame(columns=["max. Wert", "Gesamtmenge"])

        for i, part_df in enumerate(alle_tage):
            csv_filename = os.path.join(output_folder, f'Tag_{i+1}.csv')
            part_df.to_csv(csv_filename, index=False)
            max_wert = float(part_df['LKW_in_timestep'].max())
            gesamtmenge = float(part_df['LKW_in_timestep'].sum())
            gesamt_uebersicht_tage.loc[f"Woche {i+1}"] = max_wert, gesamtmenge

        # K-Means-Clustering durchführen und Cluster plotten
        plt.scatter(gesamt_uebersicht_tage['max. Wert'], gesamt_uebersicht_tage['Gesamtmenge'], color='blue',
                        label='Datenpunkte')
        cluster_array, egal = kmeans_clustering(gesamt_uebersicht_tage, anzahl_cluster_tage, 1000)
        geclusterte_tage = cluster_zuordnen(cluster_array)

        beispieltage = beispielwochen_berechnen(geclusterte_tage)
        beispielwochen_plotten(beispieltage, anzahl_cluster_tage)

    else:
        print ('Inititalisierung des Clusterns ist falsch!')

    #### CLUSTERING ENDE ####

    #### INPUT-DATEN ERZEUGEN ####

    ladewahrscheinlichkeiten_hpc = read_excel_to_df('Ladewahrscheinlichkeiten.xlsx', 'overday stops', 'H')
    ladewahrscheinlichkeiten_mcs = read_excel_to_df('Ladewahrscheinlichkeiten.xlsx', 'overday stops', 'I')
    ladewahrscheinlichkeiten_ncs = read_excel_to_df('Ladewahrscheinlichkeiten.xlsx', 'overnight stops', 'G')

    if tage_clustern:

        # LKW-Daten mit Wahrschinlichkeiten multiplizieren
        lkw_hpc = multiply_probability_with_trafficdays(beispieltage, ladewahrscheinlichkeiten_hpc)
        lkw_mcs = multiply_probability_with_trafficdays(beispieltage, ladewahrscheinlichkeiten_mcs)
        lkw_ncs = multiply_probability_with_trafficdays(beispieltage, ladewahrscheinlichkeiten_ncs)

        # auf ganze LKW runden
        lkw_hpc = rounded_dataframe_to_integer_trucks(lkw_hpc)
        lkw_mcs = rounded_dataframe_to_integer_trucks(lkw_mcs)
        lkw_ncs = rounded_dataframe_to_integer_trucks(lkw_ncs)

        dataframes = [lkw_hpc, lkw_mcs, lkw_ncs]
        lkw_hpc.name = "hpc"
        lkw_mcs.name = "mcs"
        lkw_ncs.name = "ncs"
        alle_lkw = generate_lkw_in_array(dataframes)
        dummy = 0
