#### Clustering ####
neu_clustern = False                                    # neue Daten zum Clustern erzeugen
wochen_clustern = False                                 # True, wenn Wochen geclustert werden sollen
tage_clustern = True                                    # True, wenn Tage geclustert werden sollen
jahr = 2021                                             # aus welchem Jahr stammen die Daten
anzahl_cluster_wochen = 4                               # Anzahl der Cluster für Wochenclusterung
anzahl_cluster_tage = 7                                 # Anzahl der Cluster für Tageclusterung
csv_dateiname = 'zst5651_2021.csv'                      # Dateiname der CSV-Datei
timedelta = 5                                           # Zeitauflösung in Minuten
anteil_bev = 0.049
verkehrssteigerung =  1.072
max_soc = 0.8


#### ganze LKW erzeugen ####
neue_input_daten_erzeugen = False                       # neue Inputdaten bestimmen
batterie_kapazitäten = [252, 504, 756]                  # Batteriekapazitäten
wahrscheinlichkeiten_batterien = [0.4, 0.2, 0.4]        # Verteilung Batteriekap., Reihenfolge gleich wie oben
untere_grenze_soc = 0.05                                # Anfangs-COP, untere Grenze
obere_grenze_soc = 0.3                                  # Anfangs-COP, obere Grenze
ladeleistung_liste = {'NCS': 150, 'HPC': 350, 'MCS': 1000}                   # Ladeleistung der Ladesäulen; in Reihenfolge NCS, HPC, MCS

#### Optimierung Anzahl Ladesäulen ####
investment = {'HPC': 150000, 'MCS': 350000, 'NCS': 50000}
optimierung_ladessäulenanzahl = False
wochenweise_optimieren = True
clusterweise_optimieren = False

#### Lademanagement ####
lademagement = 'kein Lademanagement'                  # wähle aus: 'kein Lademanagement', 'Durchschnitt bilden', 'Optimierung'

