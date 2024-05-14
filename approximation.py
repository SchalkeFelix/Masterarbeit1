import numpy as np
import datetime
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from main import *
from config import *
import random

# Dateiname der CSV-Datei
csv_dateiname = 'zst5651_2021.csv'

# Dateipfad zur CSV-Datei erstellen
csv_dateipfad = os.path.join(os.getcwd(), csv_dateiname)
x = start_date
y= end_date


def read_lkw_data(csv_dateipfad=csv_dateipfad, start_date = x, end_date = y):
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
    y_values = df['gesamt_LKW_R1'].to_numpy() * verkehrssteigerung
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


import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def kmeans_clustering(dataframe, num_clusters=3, max_iterations=100):
    # Daten aus dem DataFrame extrahieren
    data = dataframe.values

    # Initialisierung der Zentroiden
    centroids = initialize_centroids(data, num_clusters)

    for _ in range(max_iterations):
        # Clusterzuordnungsschritt
        cluster_labels = assign_to_clusters(data, centroids)

        # Aktualisierung der Zentroiden
        new_centroids = update_centroids(data, cluster_labels, num_clusters)

        # Prüfen auf Konvergenz
        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    # Visualisierung der Cluster
    plot_clusters(data, centroids, cluster_labels)

    # Rückgabe der zugewiesenen Cluster-Labels als Series
    return pd.Series(cluster_labels, index=dataframe.index, name='ClusterLabel')

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
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Beispiel der Verwendung:
# Angenommen 'df' ist das DataFrame, auf das die Clusteranalyse angewendet werden soll
# 'num_clusters' ist optional und bestimmt die Anzahl der zu erstellenden Cluster
# 'cluster_labels' enthält die zugeordneten Cluster-Labels für jede Zeile des DataFrames
# cluster_labels = kmeans_clustering(df, num_clusters=4)
