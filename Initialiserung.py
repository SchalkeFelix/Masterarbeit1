#### Clustering ####
neu_clustern = False                                    # neue Daten zum Clustern erzeugen
wochen_clustern = False                                 # True, wenn Wochen geclustert werden sollen
tage_clustern = True                                    # True, wenn Tage geclustert werden sollen
jahr = 2021                                             # aus welchem Jahr stammen die Daten
anzahl_cluster_wochen = 4                               # Anzahl der Cluster für Wochenclusterung
anzahl_cluster_tage = 7                                 # Anzahl der Cluster für Tageclusterung
csv_dateiname = 'zst5651_2021.csv'                      # Dateiname der CSV-Datei
timedelta = 5                                           # Zeitauflösung in Minuten

#### ganze LKW erzeugen ####
batterie_kapazitäten = [252, 504, 756]                 # Batteriekapazitäten
wahrscheinlichkeiten_batterien = [0.4, 0.2, 0.4]        # Verteilung Batteriekap., Reihenfolge gleich wie oben
untere_grenze_soc = 0.05                                # Anfangs-COP, untere Grenze
obere_grenze_soc = 0.3                                  # Anfangs-COP, obere Grenze
ladeleistung_liste = [150, 350, 1000]                   # Ladeleistung der Ladesäulen; in Reihenfolge NCS, HPC, MCS

#### Optimierung anzahl Ladesäulen ####

investionskosten_hpc = 100000                           # in €, genaue Werte noch ermitteln
investionskosten_ncs = 50000                            # in €, genaue Werte noch ermitteln
investionskosten_mcs = 150000                           # in €, genaue Werte noch ermitteln
prozentsatz_für_wartungskosten = 0.05                   # Wartungskosten werden als Prozentsatz der Investionskosten angesehen, Wert noch ermitteln