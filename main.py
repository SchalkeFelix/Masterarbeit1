import gurobipy as gp
import datetime

import matplotlib.pyplot as plt
import pandas as pd

from methods import *
from plots import *
from Initialiserung import *



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ############################################## CLUSTERING ##########################################################
    if neu_clustern:
        dummy = 0

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
            beispielwochen.to_excel('beispielwochen.xlsx', index=True)
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
            beispieltage.to_excel('beispieltage.xlsx', index=True)
            beispielwochen_plotten(beispieltage, anzahl_cluster_tage)

        else:
            print ('Inititalisierung des Clusterns ist falsch!')

    else:
        dummy = 0
        beispieltage = pd.read_excel('beispieltage.xlsx', index_col=0)
        beispielwochen = pd.read_excel('beispielwochen.xlsx', index_col=0)
        beispielwochen_plotten(beispieltage, anzahl_cluster_tage)
        dummy = 0

    ############################################## CLUSTERING ENDE ####################################################

    print('Clustern abgeschlossen!')

    ############################################## INPUT-DATEN ERZEUGEN ###############################################
    if neue_input_daten_erzeugen:
        dummy = 0
        ladewahrscheinlichkeiten_overday = read_excel_to_df('Ladewahrscheinlichkeiten.xlsx', 'overday stops', 'G')
        ladewahrscheinlichkeiten_overnight = read_excel_to_df('Ladewahrscheinlichkeiten.xlsx', 'overnight stops', 'G')

        if tage_clustern:

            # LKW-Daten mit Wahrscheinlichkeiten multiplizieren
            lkw_overday = multiply_probability_with_trafficdays(beispieltage, ladewahrscheinlichkeiten_overday)
            lkw_overnight = multiply_probability_with_trafficdays(beispieltage, ladewahrscheinlichkeiten_overnight)

            # auf ganze LKW runden
            lkw_overday = rounded_dataframe_to_integer_trucks(lkw_overday)
            lkw_overnight = rounded_dataframe_to_integer_trucks(lkw_overnight)

            # Kombinieren in einem Dataframe
            dataframes = [lkw_overnight, lkw_overday]
            lkw_overday.name = "overday"
            lkw_overnight.name = "overnight"

            # Output hier ist ein Dataframe mit Arrays für jeden LKW
            # Form jedes Arrays ist [Typ, SOC_bei_Ankunft, Batteriekapazität, max. Ladezeit, Optimierungspotential?]
            ladekurve = ladekurve()
            alle_lkw = generate_lkw_in_array(dataframes, ladekurve)
            index_list = lkw_overday.index.tolist()
            alle_lkw.index = index_list
            alle_lkw.to_excel('LKW_INPUT.xlsx', index=True)

            beispielwochen_plotten(lkw_overday, anzahl_cluster_tage)
            beispielwochen_plotten(lkw_overnight, anzahl_cluster_tage)


        if wochen_clustern:

            # Wahrscheinlichkeiten auf eine Woche hochskalieren
            ladewahrscheinlichkeiten_overday_woche = pd.concat([ladewahrscheinlichkeiten_overday] * 7, ignore_index=True)
            ladewahrscheinlichkeiten_overnight_woche = pd.concat([ladewahrscheinlichkeiten_overnight] * 7, ignore_index=True)

            # LKW-Daten mit Wahrscheinlichkeiten multiplizieren
            lkw_overday_woche = multiply_probability_with_trafficdays(beispielwochen, ladewahrscheinlichkeiten_overday_woche)
            lkw_overnight_woche = multiply_probability_with_trafficdays(beispielwochen, ladewahrscheinlichkeiten_overnight_woche)

            # auf ganze LKW runden
            lkw_overday_woche = rounded_dataframe_to_integer_trucks(lkw_overday_woche)
            lkw_overnight_woche = rounded_dataframe_to_integer_trucks(lkw_overnight_woche)

            # Kombinieren in einem Dataframe
            dataframes = [lkw_overday_woche, lkw_overnight_woche]
            lkw_overday_woche.name = "overday"
            lkw_overnight_woche.name = "overnight"

            # Output hier ist ein Dataframe mit Arrays für jeden LKW
            # Form jedes Arrays ist [Typ, SOC_bei_Ankunft, Batteriekapazität, max. Ladezeit, Optimierungspotential?]
            ladekurve = ladekurve()
            alle_lkw = generate_lkw_in_array(dataframes, ladekurve)
            index_list = lkw_overnight_woche.index.tolist()
            alle_lkw.index = index_list
            alle_lkw.to_excel('LKW_INPUT.xlsx', index=True)

    else:
        dummy = 0
        alle_lkw = pd.read_excel('LKW_INPUT.xlsx', index_col=0)


    ############################################## INPUT-DATEN ERZEUGEN ENDE ###########################################

    print('Inputdaten erzeugen abgeschlossen!')

    ############################################## ANZAHL LADESÄULEN OPTIMIERNEN #######################################


    if optimierung_ladessäulenanzahl:

        alle_lkw = pd.read_excel('LKW_INPUT.xlsx', index_col=0)
        lkw_in_tupelliste_tageweise(alle_lkw)
        lkw_in_tupelliste_wochenweise(alle_lkw)

        if wochenweise_optimieren:
            ladesäulen_anzahl_bestimmen_wochenweise(['HPC'])
            ladesäulen_anzahl_bestimmen_wochenweise(['NCS'])
            ladesäulen_anzahl_bestimmen_wochenweise(['MCS'])

        if clusterweise_optimieren:
            for i in range(0, anzahl_cluster_tage):
                ladesäulen_anzahl_bestimmen_tageweise(['HPC'], 'Cluster' + str(i))
                ladesäulen_anzahl_bestimmen_tageweise(['NCS'], 'Cluster' + str(i))
                ladesäulen_anzahl_bestimmen_tageweise(['MCS'], 'Cluster' + str(i))

    else:
       dummy = 0


