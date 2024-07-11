import datetime
from scipy.interpolate import CubicSpline
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from main import *
from config import *
from Initialiserung import *
import random
import pandas as pd
import math
import warnings
import ast
import pickle
import networkx as nx
from networkx.algorithms.flow import min_cost_flow
from networkx.algorithms.flow import max_flow_min_cost




# Dateipfad zur CSV-Datei erstellen
csv_dateipfad = os.path.join(os.getcwd(), csv_dateiname)

def read_lkw_data(csv_dateipfad=csv_dateipfad):
    # LKW-Daten aus csv einlesen und relevante Spalten summieren
    df = pd.read_csv(csv_dateipfad, delimiter=';')
    df = df[(df['Datum'] >= start_date) & (df['Datum'] <= end_date)]
    df = df.reset_index(drop=True)

    hours_difference = len(df)

    gewuenschte_spalten = ['Datum', 'Stunde', 'LoA_R1', 'Lzg_R1', 'LoA_R2', 'Lzg_R2', 'Sat_R1', 'Sat_R2']
    df = df[gewuenschte_spalten]
    df['gesamt_LKW_R1'] = df['LoA_R1'] + df['Lzg_R1'] + df['Sat_R1']
    df['gesamt_LKW_R2'] = df['LoA_R2'] + df['Lzg_R2'] + df['Sat_R2']

    # Daten für Säulendiagramm generieren (Breite: 1h, Höhe: stundenscharfer durchschnittlicher LKW-Verkehr in LKW/h)
    x_values = np.arange(hours_difference)
    y_values = df['gesamt_LKW_R1'].to_numpy() * verkehrssteigerung * anteil_bev
    Maximum = max(y_values)

    # Polynomapproximation
    # lokale Extrema anpassen um Polynom-Approximation zu verbessern
    y_values_angepasst = anpassen_liste(y_values)
    # approximierten Funktion soll durch Mitte der Säulen verlaufen
    x_values_angepasst = x_values + 0.5
    # Durchführen der Approximation
    spline = CubicSpline(x_values_angepasst, y_values_angepasst)
    x_continuous = np.linspace(0, hours_difference, int(hours_difference*60/timedelta))
    y_continuous = spline(x_continuous)

    # Zuordnen der LKW zu jedem timestep
    lkws_in_timesteps = []
    summe_in_timesteps = []
    summe = 0
    summe_lkws = 0
    for index, i in enumerate(y_continuous):
        minute = (index * timedelta) % 1440

        summe += i * timedelta/60
        wert = math.floor(summe)

        lkw_in_timestep = [wert, minute]
        lkws_in_timesteps.append(lkw_in_timestep)
        summe_lkws += wert
        summe = summe - wert
        summe_in_timesteps.append(summe)

    # Auslesen der Differenzen zwischen Ausgangsdaten und approximierten Daten (stündlich)
    summen_liste = []
    for i in range(0, len(lkws_in_timesteps), int(60 / timedelta)):
        # Summiere die ersten Werte aller Arrays im ausgewählten Bereich
        sum_of_values = sum(array[0] for array in lkws_in_timesteps[i:i + int(60 / timedelta)])
        summen_liste.append(sum_of_values)
    differenzen = [abs(a - b) for a, b in zip(y_values, summen_liste)]
    anzahl_um_10 = sum(diff > 10 for diff in differenzen)
    anzahl_um_20 = sum(diff > 20 for diff in differenzen)
    anzahl_um_50 = sum(diff > 50 for diff in differenzen)
    print(f"Differenzen größer als 10: {anzahl_um_10} Mal")
    print(f"Differenzen größer als 20: {anzahl_um_20} Mal")
    print(f"Differenzen größer als 50: {anzahl_um_50} Mal")

    # Vergleich der gesamten LKWs in einem Jahr zwischen Ausgangsdaten und approximierten Daten
    sum_y_values = np.sum(y_values)
    print(f"Summe aller LKWs in einem Jahr aus den Ausgangsdaten: {sum_y_values}")
    print(f"Summe aller LKWs in einem Jahr aus den approximierten Daten: {summe_lkws}")

    verkehrsdaten = {'x_values': x_values, 'y_values': y_values, 'x_continuous': x_continuous, 'y_continuous': y_continuous, 'differenzen': differenzen}

    return lkws_in_timesteps, verkehrsdaten, summe_lkws, Maximum

def anpassen_liste(lst):
    if len(lst) < 3:
        return lst  # Die Liste sollte mindestens drei Elemente enthalten, um Vor- und Nachgänger zu überprüfen

    angepasste_liste = [lst[0]]  # Das erste Element bleibt unverändert

    for i in range(1, len(lst)-1):
        if lst[i] > lst[i-1] and lst[i] > lst[i+1]:
            mittlere_abweichung = (lst[i-1] - 2 * lst[i] + lst[i+1]) / 2
            angepasste_liste.append(lst[i] - 0.15 * mittlere_abweichung)
        elif lst[i] < lst[i-1] and lst[i] < lst[i+1]:
            mittlere_abweichung = (lst[i-1] - 2 * lst[i] + lst[i+1]) / 2
            angepasste_liste.append(lst[i] - 0.15 * mittlere_abweichung)
        else:
            angepasste_liste.append(lst[i])

    angepasste_liste.append(lst[-1])  # Das letzte Element bleibt unverändert

    return angepasste_liste

def datumsliste_erstellen(jahr, erster_Montag):
    jahr = jahr
    erster_Montag = erster_Montag

    montage_liste = []
    sonntage_liste =[]

    # Schleife für jeden Monat im Jahr
    for monat in range(1, 13):  # von 1 bis 12 (einschließlich 12)
        tage_im_monat = 31  # wir setzen angenommen erst einmal auf 31 tage
        # je nach dem monat
        if monat == 4 or monat == 6 or monat == 9 or monat == 11:
            tage_im_monat = 30
        elif monat == 2:
            # februar hat verschiedene tage
            if (jahr % 4 == 0 and jahr % 100 != 0) or (jahr % 400 == 0):
                tage_im_monat = 29
            else:
                tage_im_monat = 28
        # Schleife für jeden Tag im aktuellen Monat
        for tag in range(1, tage_im_monat + 1):
            # Formatierung des Datums im gewünschten Stil YYMMDD
            datum_str = f"{str(jahr)[-2:]}{str(monat).zfill(2)}{str(tag).zfill(2)}"
            datum_int = int(datum_str)
            if datum_int >= erster_Montag:
                # Prüfen, ob der Tag ein Montag ist (Wochentagnummer 0 für Montag)
                if datetime.date(jahr, monat, tag).weekday() == 0:
                    montage_liste.append(datum_int)
                if datetime.date(jahr, monat, tag).weekday() == 6:
                    sonntage_liste.append(datum_int)

    print(montage_liste)
    print(sonntage_liste)

    if len(montage_liste) != len(sonntage_liste):
        montage_liste.pop()

    print(montage_liste)
    print(sonntage_liste)
    return montage_liste, sonntage_liste

def wochenweise_iterieren(erster_Montag, jahr):
    erster_Montag = erster_Montag
    montage_liste, sonntage_liste = datumsliste_erstellen(jahr, erster_Montag)
    k = 1
    sum_lkw_list = []
    max_pro_woche = []
    wochenliste = []
    for i, j in zip(montage_liste, sonntage_liste):
        timesteps, verkehrsdaten, sum_lkws, Max_pro_woche = read_lkw_data(csv_dateipfad, i, j)
        print('Woche' + str(k))

        sum_lkw_list.append(sum_lkws)
        max_pro_woche.append(Max_pro_woche)
        wochenliste.append('Woche'+str(k))
        k = k + 1
    return sum_lkw_list, max_pro_woche, wochenliste

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def initialize_centroids(data, k):
    # Zufällige Auswahl von k Datenpunkten als Startwerte für die Zentroide
    centroids_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[centroids_indices]
    return centroids

def assign_to_clusters(data, centroids):
    # Zuordnen jedes Datenpunkts zum nächsten Zentroiden
    distances = np.zeros((len(data), len(centroids)))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(data - centroid, axis=1)  # Euklidischer Abstand
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels

def update_centroids(data, cluster_labels, k):
    # Aktualisieren der Zentroiden basierend auf den zugewiesenen Clustern
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[cluster_labels == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
    return centroids

def kmeans_clustering(dataframe, num_clusters, max_iterations= 1000):
    # Daten aus dem DataFrame extrahieren
    data = dataframe.values

    # Initialisierung der Zentroiden
    centroids = initialize_centroids(data, num_clusters)


    for _ in range(max_iterations):
        # Clusterzuordnungsschritt
        cluster_labels = assign_to_clusters(data, centroids)
        # plot_clusters(data, centroids, cluster_labels)  # nur an machen für Präsi, danach löschen

        # Aktualisierung der Zentroiden
        new_centroids = update_centroids(data, cluster_labels, num_clusters)

        # Prüfen auf Konvergenz
        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    # Visualisierung der Cluster
    plot_clusters(data, centroids, cluster_labels)

    # Rückgabe der zugewiesenen Cluster-Labels als Series
    return cluster_labels, pd.Series(cluster_labels, index=dataframe.index, name='ClusterLabel')

def plot_clusters(data, centroids, cluster_labels):
    # Datenpunkte und Zentroiden plotten
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Farben für die Cluster

    # Datenpunkte plotten
    for i in range(len(centroids)):
        cluster_points = data[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1}', alpha=0.6)

    # Zentroiden plotten
    plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='x', s=100, label='Centroids')

    plt.title('K-Means Clustering')
    plt.xlabel('max. Wert')
    plt.ylabel('Gesamtmenge')
    plt.legend()
    plt.grid(True)
    plt.show()

def count_full_weeks(year):
    # Ersten Montag des Jahres finden
    first_day_of_year = date(year, 1, 1)
    first_monday = first_day_of_year + datetime.timedelta(days=(7 - first_day_of_year.weekday()))

    # Letzten Sonntag des Jahres finden
    last_day_of_year = date(year, 12, 31)
    last_sunday = last_day_of_year - datetime.timedelta(days=last_day_of_year.weekday() + 1)

    # Anzahl der Tage zwischen dem ersten Montag und dem letzten Sonntag
    days_between = (last_sunday - first_monday).days + 1

    # Anzahl der vollen Wochen berechnen
    full_weeks = days_between // 7

    return full_weeks

def first_monday_of_year(year):
    # Erstes Datum des Jahres
    first_day_of_year = datetime.date(year, 1, 1)

    # Tag der Woche für das erste Datum des Jahres (0 = Montag, 1 = Dienstag, ..., 6 = Sonntag)
    first_day_of_week = first_day_of_year.weekday()

    # Wenn der erste Tag des Jahres ein Montag ist (0), ist er der erste Montag des Jahres
    if first_day_of_week == 0:
        return 1  # Der erste Montag ist am 1. Tag des Jahres

    # Ansonsten finden wir den Tag des ersten Montags des Jahres
    # Berechnen, wie viele Tage bis zum nächsten Montag verbleiben
    days_until_next_monday = (7 - first_day_of_week) % 7

    # Tag des Jahres, an dem der erste Montag liegt
    first_monday_day_of_year = days_until_next_monday + 1

    return first_monday_day_of_year

def days_until_end_of_year(year):
    # Letztes Datum des Jahres
    last_day_of_year = datetime.date(year, 12, 31)

    # Tag der Woche für das letzte Datum des Jahres (0 = Montag, 1 = Dienstag, ..., 6 = Sonntag)
    last_day_of_week = last_day_of_year.weekday()

    # Tag des Jahres, an dem der letzte Sonntag liegt
    last_sunday = (last_day_of_year - datetime.timedelta(days=last_day_of_week + 1)).timetuple().tm_yday

    # Anzahl der Tage bis zum Jahresende (365 Tage für normale Jahre und 366 Tage für Schaltjahre)
    days_until_end = 365 if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0) else 366
    days_until_end -= last_sunday  # Anzahl der Tage vom letzten Sonntag bis zum Jahresende

    return days_until_end

def days_in_year(year):
    if (year % 4 == 0 and year % 100 != 0):
        return 366
    else:
        return 365

def gesamt_df_in_wochen_splitten(df, year):

    # nur volle Wochen betrachten, deswegen 'halbe' Wochen löschen
    mintuten_vor_erstem_Montag = (first_monday_of_year(year)-1)*24*60
    minuten_nach_letztem_sonntag = days_until_end_of_year(year)*24*60
    zeilen_drop_voher = int((mintuten_vor_erstem_Montag/5))
    zeilen_drop_nachher = int(minuten_nach_letztem_sonntag/5)
    total_rows = df.shape[0]
    rows_to_drop = list(range(zeilen_drop_voher)) + list(range(total_rows - zeilen_drop_nachher, total_rows))
    df_cleaned = df.drop(rows_to_drop)

    #in Gesamtwochen zerteilen
    num_parts = count_full_weeks(year)
    dfs = np.array_split(df_cleaned, num_parts)


    return dfs

def gesamt_df_in_tage_splitten(df, year):
    num_parts = days_in_year(year)
    dfs = np.array_split(df, num_parts)

    return dfs

def cluster_zuordnen(wochen_cluster):
    # Finde die einzigartigen Cluster-IDs
    einzigartige_cluster = set(wochen_cluster)

    # Initialisiere leere Listen für jedes Cluster
    cluster_listen = {cluster: [] for cluster in einzigartige_cluster}

    # Fülle die Listen basierend auf der Cluster-Zuordnung
    for woche, cluster in enumerate(wochen_cluster):
        cluster_listen[cluster].append(woche + 1)  # Woche + 1 für 1-basierte Indexierung

    return cluster_listen

def beispielwochen_berechnen (geclusterte_wochen_dict):
    if wochen_clustern:
        data = pd.DataFrame()
        for key in sorted(geclusterte_wochen_dict.keys()):
            current_list = geclusterte_wochen_dict[key]
            df_for_cluster = pd.DataFrame()
            i = 0
            for item in current_list:
                file_path = r'C:\Users\felix\Masterarbeit1\Wochen zu Clusterung\woche_' + str(item) + '.csv'

                # CSV-Datei einlesen
                df = pd.read_csv(file_path)
                df_for_cluster = df_for_cluster.add(df, fill_value=0)
                i += 1

            df_for_cluster = df_for_cluster.divide(i)
            df_for_cluster.rename(columns={'LKW_in_timestep': 'Cluster' + str(key)}, inplace=True)
            data['Cluster'+str(key)] =df_for_cluster['Cluster'+str(key)]

        minutenwerte = list(range(0, 7 * 24 * 60, timedelta))
        data.set_index(pd.Index(minutenwerte), inplace=True)
    if tage_clustern:
        data = pd.DataFrame()
        for key in sorted(geclusterte_wochen_dict.keys()):
            current_list = geclusterte_wochen_dict[key]
            df_for_cluster = pd.DataFrame()
            i = 0
            for item in current_list:
                file_path = r'C:\Users\felix\Masterarbeit1\Tage zu Clusterung\Tag_' + str(item) + '.csv'

                # CSV-Datei einlesen
                df = pd.read_csv(file_path)
                df_for_cluster = df_for_cluster.add(df, fill_value=0)
                i += 1

            df_for_cluster = df_for_cluster.divide(i)
            df_for_cluster.rename(columns={'LKW_in_timestep': 'Cluster' + str(key)}, inplace=True)
            data['Cluster' + str(key)] = df_for_cluster['Cluster' + str(key)]

        minutenwerte = list(range(0, 24 * 60, timedelta))
        data.set_index(pd.Index(minutenwerte), inplace=True)

    return data

def beispielwochen_plotten (beispielwochen, anzahl_cluster):

    plt.figure(figsize=(10, 5))  # Größe des Diagramms festlegen

    for i in range(0, anzahl_cluster):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        plt.plot(beispielwochen.index, beispielwochen['Cluster' + str(i)], label='Umsatz', color=colors[i], linestyle='solid')

    plt.show()

def read_excel_to_df (datei_name, mappe_name, spalten_name):
    df = pd.read_excel(datei_name, sheet_name=mappe_name, usecols=spalten_name)
    return df

def multiply_probability_with_trafficdays (beispieltage, ladewahrscheinlichkeiten):
    # Gemeinsame Indizes bestimmen
    gemeinsame_indices = ladewahrscheinlichkeiten.index.intersection(beispieltage.index)

    # Ergebnis-DataFrame initialisieren
    result = pd.DataFrame(index=gemeinsame_indices)

    df2_gemeinsam = ladewahrscheinlichkeiten.loc[gemeinsame_indices]

    # Durch alle Spalten iterieren und multiplizieren
    for column in beispieltage.columns:
        # DataFrames auf gemeinsame Indizes und aktuelle Spalte beschränken
        df1_gemeinsam1 = beispieltage.loc[gemeinsame_indices, column]
        df1_gemeinsam = df1_gemeinsam1.to_frame()
        df2_gemeinsam.columns = [column]

        # Multiplikation der Werte
        x = df1_gemeinsam.mul(df2_gemeinsam, axis=1)
        result[column] = x

    return result

def rounded_dataframe_to_integer_trucks(df):
    # Erstelle ein Ergebnis-DataFrame mit den gleichen Indizes wie df
    result_df = pd.DataFrame(index=df.index)

    # Durchlaufe jede Spalte des DataFrames
    for col in df.columns:
        i = 0  # Initialisiere den Wert i für jede Spalte
        for idx in df.index:  # Durchlaufe jeden Index der Spalte
            i += df.loc[idx, col]  # Addiere den Wert des aktuellen Index
            rounded_i = math.floor(i)  # Runde den kumulativen Wert ab
            result_df.loc[idx, col] = int(rounded_i)  # Speichere das gerundete Ergebnis im Ergebnis-DataFrame
            i -= rounded_i  # Ziehe den gerundeten Wert von i ab
    result_df = result_df.astype(int)
    return result_df

def generate_entry(df_name, ladekurve, vergebene_ids):
    # Eintrag 1: Abhängig vom Namen des DataFrames
    if "overday" in df_name:
        pausenzeit = 45
    elif "overnight" in df_name:
        pausenzeit = 480

    # Eintrag 2: Eine zufällige Zahl zwischen 0,05 und 0,3
    entry2 = round(random.uniform(untere_grenze_soc, obere_grenze_soc), 2)

    # Eintrag 3: Eine Zahl entweder 252, 504 oder 756 mit Wahrscheinlichkeiten 0,4, 0,2, 0,4
    entry3 = random.choices(batterie_kapazitäten, wahrscheinlichkeiten_batterien)[0]

    # Eintrag 4: geschätzte Ladezeit
    df_ladekurve = ladekurve
    spaltenname = entry3

    Diffenrenz_dict = {}
    ladezeit_dict = {}
    DUMMY = 0
    for key, value in ladeleistung_liste.items():
        dummy = 0
        soc = entry2
        ladezeit = 0
        dummy = 0
        while soc <= 0.8 :
            relative_geladene_energie = (df_ladekurve[spaltenname][round(soc*100, 0)] * value * (timedelta/60)) / entry3
            soc += relative_geladene_energie
            ladezeit += timedelta
            dummy = 0
        differenz = abs(ladezeit-pausenzeit)
        Diffenrenz_dict[key] = differenz
        ladezeit_dict[key] = ladezeit
        dummy = 0
    min_key = min(Diffenrenz_dict, key=Diffenrenz_dict.get)
    min_value = Diffenrenz_dict[min_key]

    entry1 = min_key
    ladezeit1 = ladezeit_dict[min_key]

    if ladezeit1 <= pausenzeit:
        entry4 = pausenzeit
        entry5 = 'Optimierungspotential'
    elif ladezeit1 > pausenzeit:
        entry4 = ladezeit1
        entry5 = 'kein Optimierungspotential'

    dummy = 0

    while True:
        entry6 = random.randint(1, 1000000)  # Beispiel ID-Bereich 1-100000
        if entry6 not in vergebene_ids:
            vergebene_ids.add(entry6)
            break

    x = [entry1, entry2, entry3, entry4, entry5, entry6]
    dummy = 0
    return [entry1, entry2, entry3, entry4, entry5, entry6]

def generate_lkw_in_array(dataframes, ladekurve):
    # Initialisieren des resultierenden DataFrames mit der gleichen Struktur wie die Eingabedaten
    result_dict = {col: [] for col in dataframes[0].columns}
    vergebene_id = set()
    # Durchlaufen der Zeilen
    num_rows = max(df.shape[0] for df in dataframes)
    for row_idx in range(0, timedelta * num_rows, timedelta):
        row_data = {col: [] for col in dataframes[0].columns}
        for df in dataframes:
            if row_idx < df.shape[0]*timedelta:
                for col in df.columns:
                    value = df.at[row_idx, col]
                    dummy = 0
                    arrays = [generate_entry(df.name, ladekurve, vergebene_id) for _ in range(value)]  # Hier wird df.name verwendet
                    row_data[col].extend(arrays)
        for col in row_data:
            result_dict[col].append(row_data[col])
        print('timestep: ' + str(row_idx))

    # Erstellen eines DataFrames aus dem Dictionary der Arrays
    result_df = pd.DataFrame(result_dict)
    return result_df

def ladekurve ():
    file_path = 'Ladekurven.xlsx'
    df_ladekurven = pd.read_excel(file_path)

    return df_ladekurven

def belegunspläne_erstellen(alle_lkw):

    df = alle_lkw

    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    print("Code executed without printing the PerformanceWarning")
    anzahl_spalten = df.shape[1]

    # Define the folder path where the files will be saved
    folder_path = 'Belegungspläne'

    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    for l in range(0, anzahl_spalten):

        new_df_HPC = pd.DataFrame(index=df.index)
        new_df_MCS = pd.DataFrame(index=df.index)
        new_df_NCS = pd.DataFrame(index=df.index)
        z = 0

        for i in range(0, len(df) * timedelta, timedelta):
            x = df['Cluster' + str(l)][i]
            data_list = ast.literal_eval(x)
            print(i)
            for j in data_list:
                z += 1
                y = j[3]
                q = j[0]
                col_name = 'col' + str(z)  # Create a unique column name

                if q == 'HPC':
                    # Initialize the new column with zeros
                    new_df_HPC[col_name] = 0

                    for k in range(i, i + y, timedelta):
                        new_df_HPC.at[k, col_name] = 1

                if q == 'MCS':
                    # Initialize the new column with zeros
                    new_df_MCS[col_name] = 0

                    for k in range(i, i + y, timedelta):
                        new_df_MCS.at[k, col_name] = 1
                if q == 'NCS':
                    # Initialize the new column with zeros
                    new_df_NCS[col_name] = 0

                    for k in range(i, i + y, timedelta):
                        new_df_NCS.at[k, col_name] = 1

        dummy = 0
        new_df_HPC.to_excel(os.path.join(folder_path, 'Belegungsplan_HPC_Cluster' + str(l) + '.xlsx'), index=True)
        print('HPC gespeichert')
        new_df_MCS.to_excel(os.path.join(folder_path, 'Belegungsplan_MCS_Cluster' + str(l) + '.xlsx'), index=True)
        print('MCS gespeichert')
        new_df_NCS.to_excel(os.path.join(folder_path, 'Belegungsplan_NCS_Cluster' + str(l) + '.xlsx'), index=True)
        print('NCS gespeichert')
        print('Cluster' + str(l) + ' gespeichert!')

def belegungspläne_array(alle_lkw):

    df = alle_lkw

    anzahl_spalten = df.shape[1]

    liste_cluster = []
    for i in range(0, anzahl_spalten):
        liste_cluster.append('Cluster' + str(i))
    liste_ladesäulen = ['NCS', 'HPC', 'MCS']
    # Dictionary zum Speichern der Ergebnisse
    result = {}

    # Über die Spalten iterieren

    for l in range(0, anzahl_spalten):
        cluster_name = 'Cluster' + str(l)
        for i in range(0, len(df) * timedelta, timedelta):
            x = df[cluster_name][i]
            data_list = ast.literal_eval(x)
            print(i)
            for j in data_list:
                ladesaeulentyp = j[0]
                ladezeit = j[3]
                truck_id = j[5]
                dummy = 0

                ankunftszeit = i
                abfahrtszeit = i + ladezeit

                if cluster_name not in result:
                    result[cluster_name] = {}

                if ladesaeulentyp not in result[cluster_name]:
                    result[cluster_name][ladesaeulentyp] = []

                result[cluster_name][ladesaeulentyp].append((truck_id, ankunftszeit, abfahrtszeit))
            print(i)
    dummY = 0
    print(result)
    # Liste der Ereignisse
    events = []
    # Define the folder path where the files will be saved
    folder_path = 'Belegungspläne_Array'

    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)
    for l in liste_cluster:
        for k in liste_ladesäulen:
            data = result[l][k]
            # Ereignisse aus den Daten extrahieren
            for truck in data:
                lkw_id, arrival, departure = truck
                events.append((arrival, 'Ankunft', lkw_id))
                events.append((departure, 'Abfahrt', lkw_id))

            # Ereignisse sortieren
            events.sort()

            # Ladesäulen-Dict initialisieren
            charging_stations = {}
            lkw_to_station = {}

            # Zähler für die maximale Anzahl gleichzeitiger Ladesäulen
            max_stations_needed = 0
            current_stations = 0
            available_stations = []

            # Anzahl der LKWs und Grenze für 80%
            total_trucks = len(data)
            assigned_trucks = 0
            limit = int(total_trucks * 0.8)

            # Durch die Ereignisse gehen
            for event in events:
                time, event_type, lkw_id = event
                if assigned_trucks >= limit:
                    break
                if event_type == 'Ankunft':
                    current_stations += 1
                    if available_stations:
                        station = available_stations.pop()
                    else:
                        station = max_stations_needed + 1
                    charging_stations[station] = time
                    lkw_to_station[lkw_id] = station
                    assigned_trucks += 1
                    if current_stations > max_stations_needed:
                        max_stations_needed = current_stations
                else:
                    current_stations -= 1
                    if lkw_id in lkw_to_station:
                        station = lkw_to_station[lkw_id]
                        available_stations.append(station)
                        del charging_stations[station]
            pickle_datei = os.path.join(folder_path, l + '_' + k + '.pkl')
            pickle_datei1 = os.path.join(folder_path, l + '_' + k + '_max.pkl')
            with open(pickle_datei, 'wb') as f:
                pickle.dump(lkw_to_station, f)
            with open(pickle_datei1, 'wb') as f:
                pickle.dump(max_stations_needed, f)

def lkw_in_tupelliste_tageweise(alle_lkw):

    df = alle_lkw

    anzahl_spalten = df.shape[1]

    # Define the folder path where the files will be saved
    folder_path = 'Tupellisten_LKW'

    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Dictionary zum Speichern der Ergebnisse

    größte_abfahrtszeit = 0
    # Über die Spalten iterieren

    for l in range(0, anzahl_spalten):
        result = []
        cluster_name = 'Cluster' + str(l)
        for i in range(0, len(df) * timedelta, timedelta):
            x = df[cluster_name][i]
            data_list = ast.literal_eval(x)
            print(i)
            for j in data_list:
                ladesaeulentyp = j[0]
                ladezeit = j[3]
                truck_id = j[5]
                dummy = 0

                ankunftszeit = i
                abfahrtszeit = i + ladezeit
                abfahrtszeit_int = int(abfahrtszeit/timedelta)
                if abfahrtszeit_int >= größte_abfahrtszeit:
                    größte_abfahrtszeit = abfahrtszeit_int
                x = (ladesaeulentyp, int(ankunftszeit/timedelta), int(abfahrtszeit/timedelta), truck_id)
                result.append((ladesaeulentyp, int(ankunftszeit/timedelta), int(abfahrtszeit/timedelta), truck_id))
        pickle_datei = os.path.join(folder_path, 'Tupelliste_' + cluster_name + '.pkl')
        pickle_datei1 = os.path.join(folder_path, 'Tupelliste_' + cluster_name + '_groeßte_Abfahrt.pkl')
        with open(pickle_datei, 'wb') as f:
            pickle.dump(result, f)
        with open(pickle_datei1, 'wb') as f:
            pickle.dump(größte_abfahrtszeit, f)

def lkw_in_tupelliste_wochenweise(alle_lkw):

    df = alle_lkw

    anzahl_spalten = df.shape[1]

    # Define the folder path where the files will be saved
    folder_path = 'Tupellisten_LKW'

    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Dictionary zum Speichern der Ergebnisse

    größte_abfahrtszeit = 0
    result = []
    # Über die Spalten iterieren

    for l in range(0, anzahl_spalten):

        cluster_name = 'Cluster' + str(l)
        for i in range(0, len(df) * timedelta, timedelta):
            x = df[cluster_name][i]
            data_list = ast.literal_eval(x)
            print(i)
            for j in data_list:
                ladesaeulentyp = j[0]
                ladezeit = j[3]
                truck_id = j[5]
                dummy = 0

                ankunftszeit = int(i+(1440*l))
                abfahrtszeit = int(i + ladezeit+(1440*l))
                abfahrtszeit_int = int(abfahrtszeit/timedelta)
                dummy = 0
                if abfahrtszeit_int >= größte_abfahrtszeit:
                    größte_abfahrtszeit = abfahrtszeit_int
                x = (ladesaeulentyp, int(ankunftszeit/timedelta), int(abfahrtszeit/timedelta), truck_id)
                result.append((ladesaeulentyp, int(ankunftszeit/timedelta), int(abfahrtszeit/timedelta), truck_id))
    pickle_datei = os.path.join(folder_path, 'Tupelliste_Woche.pkl')
    pickle_datei1 = os.path.join(folder_path, 'Tupelliste_Woche_groeßte_Abfahrt.pkl')
    with open(pickle_datei, 'wb') as f:
        pickle.dump(result, f)
    with open(pickle_datei1, 'wb') as f:
        pickle.dump(größte_abfahrtszeit, f)

def tupelliste_sortieren(cluster_verteilung_kumuliert):
    k = 0
    result = []
    for i in range(0, len(cluster_verteilung_kumuliert)):
        with open('Tupellisten_LKW/Tupelliste_Cluster' + str(i) +'.pkl', 'rb') as file:
            daten = pickle.load(file)
        for j in range(k, cluster_verteilung_kumuliert[i]):
            for tupel in daten:
                Ladesäule = tupel[0]
                Ankunft = int(tupel[1]+((1440/timedelta)*j))
                Abfahrt = int(tupel[2]+((1440/timedelta)*j))
                dummy = 0
                if Abfahrt <= int((365*24*60)/timedelta):
                    result.append((Ladesäule, Ankunft, Abfahrt))
            dummy = 0
        k = cluster_verteilung_kumuliert[i]
        print('Cluster ' + str(i) + ' wurde hinzugefügt!' )
    return result

def ladesäulen_anzahl_bestimmen_wochenweise(ladesäulentyp):

    counts = {
        'NCS': 0,
        'HPC': 0,
        'MCS': 0
    }
    lkw_gesamt = 0
    df = pd.read_excel('LKW_INPUT.xlsx', index_col=0)
    for l in range(0, 6):
        for i in range(0, len(df) * timedelta, timedelta):
            x = df['Cluster' + str(l)][i]
            data_list = ast.literal_eval(x)
            dummy = 0
            for j in data_list:
                typ = j[0]
                if typ in counts:
                    counts[typ] += 1
                lkw_gesamt += 1
                dummy = 0

    print(counts)
    print(lkw_gesamt)
    with open('Tupellisten_LKW/Tupelliste_Woche.pkl', 'rb') as file:
        trucks1 = pickle.load(file)
    with open('Tupellisten_LKW/Tupelliste_Woche_groeßte_Abfahrt.pkl', 'rb') as file:
        größte_Abfahrtszeit = pickle.load(file)

    weights = {
        "NCS": 0,  # Gewicht für Typ 1
        "MCS": 0,  # Gewicht für Typ 2
        "HPC": 0}
    trucks = []
    for tupel in trucks1:
        typ, ankunftszeit, abfahrtszeit, truck_id = tupel
        if typ == ladesäulentyp[0]:
            x = (typ, ankunftszeit, abfahrtszeit, truck_id)
            trucks.append(x)

    print('Liste wurde aussortiert!')

    # Initialisieren des Graphen
    G = nx.DiGraph()

    # Anzahl der Zeitabschnitte (5 Minuten Intervalle in einem Tag)
    time_intervals = größte_Abfahrtszeit
    dummy = 0

    # Start- und Endknoten
    start_node = 'start'
    end_node = 'end'

    # Füge den Start- und Endknoten hinzu
    G.add_node(start_node)
    G.add_node(end_node)

    # Füge Knoten für jeden Zeitpunkt für jeden Ladesäulentyp hinzu
    charging_types = ladesäulentyp
    for typ in charging_types:
        for t in range(time_intervals):
            G.add_node(f"{typ}_{t}")

    # Füge Kanten mit hohen Kosten zwischen aufeinanderfolgenden Zeitknoten für jeden Ladesäulentyp hinzu
    low_cost = 0  # Sehr niedrige Kosten
    high_cost = 10e6
    for typ in charging_types:
        for t in range(time_intervals - 1):
            weight = weights[typ]
            G.add_edge(f"{typ}_{t}", f"{typ}_{t + 1}", weight=low_cost, capacity=float('inf'))

    # Füge Kanten von Start zu den ersten Zeitknoten jedes Ladesäulentyps hinzu
    for typ in charging_types:
        G.add_edge(start_node, f"{typ}_0", weight=0, capacity=float('inf'))

    # Füge Kanten von den letzten Zeitknoten jedes Ladesäulentyps zum Endknoten hinzu
    for typ in charging_types:
        G.add_edge(f"{typ}_{time_intervals - 1}", end_node, weight=0, capacity=float('inf'))

    print('Grundaufbau des Graphen steht!')

    # Füge Kanten für die LKWs hinzu
    truck_edges = {}
    for typ, start_time, end_time, truck_id in trucks:
        edge = (f"{typ}_{start_time}", f"{typ}_{end_time}")
        G.add_edge(*edge, weight=-1, capacity=1)
        truck_edges[edge] = truck_id

    print('LKW-Kanten wurden hinzugefügt!')

    # Funktion zur Berechnung des minimalen Kostenflusses
    def calculate_min_cost_flow(graph, required_flow):
        # Füge eine Superquelle und ein Superziel hinzu
        super_source = 'super_source'
        super_sink = 'super_sink'
        graph.add_node(super_source)
        graph.add_node(super_sink)

        # Verbinde die Superquelle mit dem Startknoten
        graph.add_edge(super_source, start_node, weight=0, capacity=required_flow)

        # Verbinde das Endknoten mit dem Superziel
        graph.add_edge(end_node, super_sink, weight=0, capacity=required_flow)

        # Berechne den maximalen Fluss mit minimalen Kosten
        flow_dict = max_flow_min_cost(graph, super_source, super_sink)

        # Entferne die Superquelle und das Superziel
        graph.remove_node(super_source)
        graph.remove_node(super_sink)

        return flow_dict

    flow = 1
    while True:
        flow_dict = calculate_min_cost_flow(G, flow)
        used_truck_edges = sum(
            flow_dict[f"{typ}_{start_time}"][f"{typ}_{end_time}"] for typ, start_time, end_time, _ in trucks)
        threshold = 0.8 * counts[ladesäulentyp[0]]
        if used_truck_edges >= threshold:
            break
        else:
            print('Anzahl Ladesäulen von ' + str(flow) + ' war nicht ausreichend! (Ladequote: ' + str(
                round(used_truck_edges / counts[ladesäulentyp[0]], 4) * 100) + ' %)')

            flow += 1

    # Drucke den Fluss für jede Kante aus
    loaded_trucks = []
    for u in flow_dict:
        for v in flow_dict[u]:
            flow_value = flow_dict[u][v]
            if flow_value > 0:
                print(f"Fluss von {u} nach {v}: {flow_value}")
                if (u, v) in truck_edges and flow_value > 0:
                    loaded_trucks.append(truck_edges[(u, v)])

    print('Anzahl Ladesäulen sind: ' + str(flow))
    print('Geladene LKWs:', loaded_trucks)

    folder_path = 'Geladene_LKW'
    os.makedirs(folder_path, exist_ok=True)
    pickle_datei2 = os.path.join(folder_path, 'geladene_LKW_'+ ladesäulentyp[0] + '_wochenweise.pkl')
    with open(pickle_datei2, 'wb') as f:
        pickle.dump(loaded_trucks, f)

def ladesäulen_anzahl_bestimmen_tageweise(ladesäulentyp, clustername):

    counts = {
        'NCS': 0,
        'HPC': 0,
        'MCS': 0
    }
    lkw_gesamt = 0
    df = pd.read_excel('LKW_INPUT.xlsx', index_col=0)
    for i in range(0, len(df) * timedelta, timedelta):
        x = df[clustername][i]
        data_list = ast.literal_eval(x)
        dummy = 0
        for j in data_list:
            typ = j[0]
            if typ in counts:
                counts[typ] += 1
            lkw_gesamt += 1
            dummy = 0

    print(counts)
    print(lkw_gesamt)
    with open('Tupellisten_LKW/Tupelliste_' + clustername + '.pkl', 'rb') as file:
        trucks1 = pickle.load(file)
    with open('Tupellisten_LKW/Tupelliste_'+ clustername +'_groeßte_Abfahrt.pkl', 'rb') as file:
        größte_Abfahrtszeit = pickle.load(file)

    weights = {
        "NCS": 0,  # Gewicht für Typ 1
        "MCS": 0,  # Gewicht für Typ 2
        "HPC": 0}
    trucks = []
    for tupel in trucks1:
        typ, ankunftszeit, abfahrtszeit, truck_id = tupel
        if typ == ladesäulentyp[0]:
            x = (typ, ankunftszeit, abfahrtszeit, truck_id)
            trucks.append(x)

    print('Liste wurde aussortiert!')

    # Initialisieren des Graphen
    G = nx.DiGraph()

    # Anzahl der Zeitabschnitte (5 Minuten Intervalle in einem Tag)
    time_intervals = größte_Abfahrtszeit
    dummy = 0

    # Start- und Endknoten
    start_node = 'start'
    end_node = 'end'

    # Füge den Start- und Endknoten hinzu
    G.add_node(start_node)
    G.add_node(end_node)

    # Füge Knoten für jeden Zeitpunkt für jeden Ladesäulentyp hinzu
    charging_types = ladesäulentyp
    for typ in charging_types:
        for t in range(time_intervals):
            G.add_node(f"{typ}_{t}")

    # Füge Kanten mit hohen Kosten zwischen aufeinanderfolgenden Zeitknoten für jeden Ladesäulentyp hinzu
    low_cost = 0  # Sehr niedrige Kosten
    high_cost = 10e6
    for typ in charging_types:
        for t in range(time_intervals - 1):
            weight = weights[typ]
            G.add_edge(f"{typ}_{t}", f"{typ}_{t + 1}", weight=high_cost, capacity=float('inf'))

    # Füge Kanten von Start zu den ersten Zeitknoten jedes Ladesäulentyps hinzu
    for typ in charging_types:
        G.add_edge(start_node, f"{typ}_0", weight=0, capacity=float('inf'))

    # Füge Kanten von den letzten Zeitknoten jedes Ladesäulentyps zum Endknoten hinzu
    for typ in charging_types:
        G.add_edge(f"{typ}_{time_intervals - 1}", end_node, weight=0, capacity=float('inf'))

    print('Grundaufbau des Graphen steht!')

    # Füge Kanten für die LKWs hinzu
    truck_edges = {}
    for typ, start_time, end_time, truck_id in trucks:
        edge = (f"{typ}_{start_time}", f"{typ}_{end_time}")
        G.add_edge(*edge, weight=0, capacity=1)
        truck_edges[edge] = truck_id

    print('LKW-Kanten wurden hinzugefügt!')

    # Funktion zur Berechnung des minimalen Kostenflusses
    def calculate_min_cost_flow(graph, required_flow):
        # Füge eine Superquelle und ein Superziel hinzu
        super_source = 'super_source'
        super_sink = 'super_sink'
        graph.add_node(super_source)
        graph.add_node(super_sink)

        # Verbinde die Superquelle mit dem Startknoten
        graph.add_edge(super_source, start_node, weight=0, capacity=required_flow)

        # Verbinde das Endknoten mit dem Superziel
        graph.add_edge(end_node, super_sink, weight=0, capacity=required_flow)

        # Berechne den maximalen Fluss mit minimalen Kosten
        flow_dict = max_flow_min_cost(graph, super_source, super_sink)

        # Entferne die Superquelle und das Superziel
        graph.remove_node(super_source)
        graph.remove_node(super_sink)

        return flow_dict

    flow = 1
    while True:
        flow_dict = calculate_min_cost_flow(G, flow)
        used_truck_edges = sum(
            flow_dict[f"{typ}_{start_time}"][f"{typ}_{end_time}"] for typ, start_time, end_time, _ in trucks)
        threshold = 0.8 * counts[ladesäulentyp[0]]
        if used_truck_edges >= threshold:
            break
        else:
            print('Anzahl Ladesäulen von ' + str(flow) + ' war nicht ausreichend! (Ladequote: ' + str(
                round(used_truck_edges / counts[ladesäulentyp[0]], 4) * 100) + ' %)')

            flow += 1

    # Drucke den Fluss für jede Kante aus
    loaded_trucks = []
    for u in flow_dict:
        for v in flow_dict[u]:
            flow_value = flow_dict[u][v]
            if flow_value > 0:
                print(f"Fluss von {u} nach {v}: {flow_value}")
                if (u, v) in truck_edges and flow_value > 0:
                    loaded_trucks.append(truck_edges[(u, v)])

    print('Anzahl Ladesäulen sind: ' + str(flow))
    print('Geladene LKWs:', loaded_trucks)

    folder_path = 'Geladene_LKW'
    os.makedirs(folder_path, exist_ok=True)
    pickle_datei2 = os.path.join(folder_path, 'geladene_LKW_'+ ladesäulentyp[0] + '_' + clustername + '_tageweise.pkl')
    with open(pickle_datei2, 'wb') as f:
        pickle.dump(loaded_trucks, f)

def lastgang_plotten (beispielwochen, anzahl_cluster):

    plt.figure(figsize=(10, 5))  # Größe des Diagramms festlegen

    for i in range(0, anzahl_cluster):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        plt.plot(beispielwochen.index, beispielwochen['Last_Cluster' + str(i)], label='Umsatz', color=colors[i], linestyle='solid')

    plt.show()

