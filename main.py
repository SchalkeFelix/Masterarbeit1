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

        with open('Geladene_LKW/geladene_LKW_HPC_wochenweise.pkl', 'rb') as file:
            geladene_lkw_hpc = pickle.load(file)
        with open('Geladene_LKW/geladene_LKW_MCS_wochenweise.pkl', 'rb') as file:
            geladene_lkw_mcs = pickle.load(file)
        with open('Geladene_LKW/geladene_LKW_NCS_wochenweise.pkl', 'rb') as file:
            geladene_lkw_ncs = pickle.load(file)


        print('Bestimmung der Anzahl der Ladesäulen abgeschlossen!')
    else:

        with open('Geladene_LKW/geladene_LKW_HPC_wochenweise.pkl', 'rb') as file:
            geladene_lkw_hpc = pickle.load(file)
        with open('Geladene_LKW/geladene_LKW_MCS_wochenweise.pkl', 'rb') as file:
            geladene_lkw_mcs = pickle.load(file)
        with open('Geladene_LKW/geladene_LKW_NCS_wochenweise.pkl', 'rb') as file:
            geladene_lkw_ncs = pickle.load(file)

        print('Bestimmung der Anzahl der Ladesäulen abgeschlossen!')

    ######################################### ANZAHL LADESÄULEN OPTIMIERNEN ENDE #######################################

    ################################################# LADEMANAGEMENT ###################################################

    alle_geladenen_lkw = geladene_lkw_hpc + geladene_lkw_ncs + geladene_lkw_mcs
    counter = 0
    indices = list(range(0, 2001, timedelta))
    df_ladekurve = ladekurve()
    result_df_lastkurve = pd.DataFrame(0, columns=['Last_Cluster0', 'Last_Cluster1', 'Last_Cluster2', 'Last_Cluster3', 'Last_Cluster4', 'Last_Cluster5', 'Last_Cluster6'], index = indices)
    for element in alle_geladenen_lkw:
        anzahl_spalten = alle_lkw.shape[1]
        for l in range(0, anzahl_spalten):
            cluster_name = 'Cluster' + str(l)
            for i in range(0, len(alle_lkw) * timedelta, timedelta):
                x = alle_lkw[cluster_name][i]
                data_list = ast.literal_eval(x)
                dummy = 0
                for j in data_list:
                    if element == j[5]:
                        if j[4] == 'kein Optimierungspotential' :
                            soc = j[1]
                            kapazität = j[2]
                            timestep = i
                            ladeleistung = ladeleistung_liste[j[0]]

                            while soc <= 0.8:
                                relative_geladene_energie = (df_ladekurve[kapazität][round(soc * 100, 0)] * ladeleistung * (
                                            timedelta / 60)) / kapazität
                                soc += relative_geladene_energie
                                alter_wert = result_df_lastkurve.at[timestep, 'Last_Cluster'+str(l)]
                                neuer_wert = alter_wert + (df_ladekurve[kapazität][round(soc * 100, 0)]*ladeleistung)
                                result_df_lastkurve.at[timestep, 'Last_Cluster'+str(l)] = neuer_wert
                                dummy = 0
                                timestep += timedelta

        counter += 1
        print(counter/len(alle_geladenen_lkw))
    dummy = 0