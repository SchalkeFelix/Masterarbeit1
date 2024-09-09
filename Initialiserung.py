#### Clustering ####
neu_clustern = False                                    # neue Daten zum Clustern erzeugen
wochen_clustern = False                                 # True, wenn Wochen geclustert werden sollen
tage_clustern = True                                    # True, wenn Tage geclustert werden sollen
jahr = 2021                                             # aus welchem Jahr stammen die Daten
anzahl_cluster_wochen = 4                               # Anzahl der Cluster für Wochenclusterung
anzahl_cluster_tage = 7                                 # Anzahl der Cluster für Tageclusterung
csv_dateiname = 'zst5651_2021.csv'                      # Dateiname der CSV-Datei
timedelta = 5                                           # Zeitauflösung in Minuten
anteil_bev = 0.049                                      # 3 Nachkommastellen
verkehrssteigerung = 1.072                             # 3 Nachkommastellen
max_soc = 0.8
ladeqoute = 0.8


#### ganze LKW erzeugen ####
neue_input_daten_erzeugen = False                       # neue Inputdaten bestimmen
batterie_kapazitäten = [252, 504, 756]                  # Batteriekapazitäten
wahrscheinlichkeiten_batterien = [0.6025, 0.3065, 0.0910]        # Verteilung Batteriekap., Reihenfolge gleich wie oben
untere_grenze_soc = 0.05                                # Anfangs-COP, untere Grenze
obere_grenze_soc = 0.3                                  # Anfangs-COP, obere Grenze
ladeleistung_liste = {'NCS': 88, 'HPC': 350, 'MCS': 1000}  # Ladeleistung der Ladesäulen; in Reihenfolge NCS, HPC, MCS
anteil_nicht_depot_overday = 0.5
anteil_nicht_depot_overnight = 0.2

#### Optimierung Anzahl Ladesäulen ####
# investment = {'HPC': 150000, 'MCS': 350000, 'NCS': 50000}
optimierung_ladessäulenanzahl = True
wochenweise_optimieren = True
clusterweise_optimieren = False

#### Lademanagement ####
lademagement = 'Durchschnitt bilden'                 # wähle aus: 'kein Lademanagement', 'Durchschnitt bilden', 'Optimierung'

