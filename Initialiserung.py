#### Clustering ####
neu_clustern = False                                    # neue Daten zum Clustern erzeugen
wochen_clustern = True                                  # True, wenn Wochen geclustert werden sollen
tage_clustern = False                                   # True, wenn Tage geclustert werden sollen
jahr = 2021                                             # aus welchem Jahr stammen die Daten
anzahl_cluster_wochen = 4                               # Anzahl der Cluster für Wochenclusterung
anzahl_cluster_tage = 7                                 # Anzahl der Cluster für Tageclusterung
csv_dateiname = 'zst5651_2021.csv'                      # Dateiname der CSV-Datei
timedelta = 5                                           # Zeitauflösung in Minuten

#### ganze LKW erzeugen ####
batterie_kapazitäten = [252, 504, 756]
wahrscheinlichkeiten_batterien = [0.4, 0.2, 0.4]
untere_grenze_soc = 0.05
obere_grenze_soc = 0.3