import gurobipy as gp
from gurobipy import GRB
from Initialiserung import*
from methods import*
import ast
import pickle
import time
from heapq import heappush, heappop

'''
# Erstellen des Modells
model = gp.Model("Optimierung Anzahl Ladesäulen")

# Anzahl der Entscheidungsvariablen pro Array und Anzahl der Arrays
num_vars_per_array = 4  # [Anzahl_HPC, Anzahl_MCS, Anzahl_NCS, Strommenge]
if wochen_clustern:
    num_arrays = int(10080 / timedelta)
elif tage_clustern:
    num_arrays = int(1440 / timedelta)

# Hinzufügen von Entscheidungsvariablen
# Variablen: x_hpc, x_mcs, x_ncs, strommenge für jedes der Arrays
vars = model.addVars(num_arrays, num_vars_per_array, name="x", vtype=GRB.CONTINUOUS, lb=0)

# Hinzufügen von anderen Variabeln
# Hinzufügen der neuen Variablen y, Anzahl gleichzeitig ladender LKW
y_vars = model.addVars(num_arrays, name="y", vtype=GRB.CONTINUOUS, lb=0)

# Hinzufügen der neuen Variable z, Anzahl geladener LKW insgesamt
z_var = model.addVar(name="z", vtype=GRB.CONTINUOUS, lb=0)

# Hinzufügen der binären Variablen für die Bedingung "LKW fährt an Ladesäule"
lkw_an_ladesaeule = model.addVars(num_arrays, name="lkw_an_ladesaeule", vtype=GRB.BINARY)


# Variablen für den letzten Array
letzter_array = num_arrays-1
x_hpc = vars[letzter_array, 0]
x_mcs = vars[letzter_array, 1]
x_ncs = vars[letzter_array, 2]
strommenge = vars[letzter_array, 3]         # evtl. brauch man das hier nicht, wenn Berechnung in gesamter

# Setzen der Zielfunktion
model.setObjective(
    x_hpc * (1 + prozentsatz_für_wartungskosten) * investionskosten_hpc +
    x_mcs * (1 + prozentsatz_für_wartungskosten) * investionskosten_mcs +
    x_ncs * (1 + prozentsatz_für_wartungskosten) * investionskosten_ncs +
    strommenge,
    GRB.MINIMIZE
)

#### Nebenbedingungen

for t in range(1, num_arrays):

    # Anzahl der Ladesäulen darf nicht kleiner werden
    model.addConstr(vars[t-1, 0] <= vars[t, 0], name=f"nb_hpc_{t}")
    model.addConstr(vars[t - 1, 1] <= vars[t, 1], name=f"nb_mcs_{t}")
    model.addConstr(vars[t - 1, 2] <= vars[t, 2], name=f"nb_ncs_{t}")

    # ladende LKW dürfen Anzahl der Ladesäulen nicht überschreiten

    model.addConstr(
        vars[t, 0] + vars[t, 1] + vars[t, 2] >= y_vars[t],
        name=f"nb_sum_{t}")

    dummy = 0
'''
'''
import pandas as pd
import numpy as np
import random
# Number of rows
n_rows = 100

# Indexes
indices = np.arange(0, 5 * n_rows, 5)

# Possible values for the first entry
first_entry_values = ['HPC', 'MCS', 'NCS']

# Possible values for the third entry
third_entry_values = [252, 504, 756]

# Generate the DataFrame
data = []
for _ in range(n_rows):
    x= random.random()
    row = [
        np.array([
            np.random.choice(first_entry_values),        # First entry
            round(np.random.uniform(0.05, 0.3), 2),     # Second entry
            np.random.choice(third_entry_values),        # Third entry
            np.random.choice(np.arange(10, 40, 5)),      # Fourth entry
            'optimierungspotential'                      # Fifth entry
        ])
    ]
    data.append(row)


# Create the DataFrame
df = pd.DataFrame(data, index=indices, columns=['Data'])
df.head()

# Extract the fourth entry and compute the new index value
fourth_entries = df['Data'].apply(lambda x: int(x[3]))
new_indices = df.index + fourth_entries-timedelta

# Create a new DataFrame with the same indices as the original
new_df = pd.DataFrame(0, index=indices, columns=[f'Array_{i}' for i in range(n_rows)])

# Populate the new DataFrame
for i, idx in enumerate(df.index):
    new_idx = new_indices[idx]
    new_df.loc[idx:new_idx, f'Array_{i}'] = 1

print("\nNew DataFrame:")
print(new_df.head(20))
'''
"""
import warnings

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

    for i in range(0, len(df)*timedelta, timedelta):
        x = df['Cluster'+str(l)][i]
        data_list = ast.literal_eval(x)
        print(i)
        for j in data_list:
            z+=1
            y = j[3]
            q = j[0]
            col_name = 'col'+str(z)  # Create a unique column name

            if q == 'HPC':
                # Initialize the new column with zeros
                new_df_HPC[col_name] = 0

                for k in range(i, i+y, timedelta):
                    new_df_HPC.at[k, col_name] = 1

            if q == 'MCS':
                # Initialize the new column with zeros
                new_df_MCS[col_name] = 0

                for k in range(i, i+y, timedelta):
                    new_df_MCS.at[k, col_name] = 1
            if q == 'NCS':
                # Initialize the new column with zeros
                new_df_NCS[col_name] = 0

                for k in range(i, i+y, timedelta):
                    new_df_NCS.at[k, col_name] = 1

    dummy = 0
    new_df_HPC.to_excel(os.path.join(folder_path, 'Belegungsplan_HPC_Cluster' + str(l) + '.xlsx'), index=True)
    print('HPC gespeichert')
    new_df_MCS.to_excel(os.path.join(folder_path, 'Belegungsplan_MCS_Cluster' + str(l) + '.xlsx'), index=True)
    print('MCS gespeichert')
    new_df_NCS.to_excel(os.path.join(folder_path, 'Belegungsplan_NCS_Cluster' + str(l) + '.xlsx'), index=True)
    print('NCS gespeichert')
    print('Cluster'+str(l)+' gespeichert!')
dummy = 0


import pandas as pd
import pulp

dummy = 0
HPC_Cluster0 = pd.read_excel('Belegungspläne/Belegungsplan_HPC_Cluster0.xlsx', index_col=0)
max_index_to_keep = 1435

# Index-Werte ermitteln, die kleiner oder gleich max_index_to_keep sind
indices_to_keep = HPC_Cluster0.index[HPC_Cluster0.index <= max_index_to_keep]
dummy = 0

df = HPC_Cluster0.loc[indices_to_keep]
dummy = 0

# Anzahl der LKWs und Zeitschritte
num_lkw = df.shape[1]
num_steps = df.shape[0]

# Setze das Optimierungsproblem
prob = pulp.LpProblem("Ladesaeulen_Optimierung", pulp.LpMinimize)

# Variablen: y[j] ist 1, wenn Ladesäule j genutzt wird
y = pulp.LpVariable.dicts("y", range(num_lkw), 0, 1, pulp.LpBinary)

# Variablen: x[i][j] ist 1, wenn LKW i an Ladesäule j geladen wird
x = pulp.LpVariable.dicts("x", [(i, j) for i in range(num_lkw) for j in range(num_lkw)], 0, 1, pulp.LpBinary)

# Ziel: Minimierung der Anzahl der genutzten Ladesäulen
prob += pulp.lpSum(y[j] for j in range(num_lkw))

# Nebenbedingungen: Jeder LKW darf an höchstens einer Ladesäule geladen werden
for i in range(num_lkw):
    prob += pulp.lpSum(x[i, j] for j in range(num_lkw)) <= 1

# Nebenbedingungen: Keine Überschneidung der Ladezeiten an einer Ladesäule
for j in range(num_lkw):
    for t in range(num_steps):
        prob += pulp.lpSum(x[i, j] * df.iloc[t, i] for i in range(num_lkw)) <= y[j]

# Nebenbedingungen: Mindestens 80% der LKWs müssen geladen werden
prob += pulp.lpSum(x[i, j] for i in range(num_lkw) for j in range(num_lkw)) >= 0.8 * num_lkw

# Ladesäulen können nur genutzt werden, wenn ein LKW daran geladen wird
for j in range(num_lkw):
    for i in range(num_lkw):
        prob += x[i, j] <= y[j]

# Problem lösen
prob.solve()

# Ergebnis extrahieren
solution = {}
for i in range(num_lkw):
    for j in range(num_lkw):
        if pulp.value(x[i, j]) == 1:
            if j in solution:
                solution[j].append(i)
            else:
                solution[j] = [i]

# Ausgabe der Lösung
used_stations = sorted(k for k in solution.keys() if pulp.value(y[k]) == 1)
print("Anzahl benötigter Ladesäulen:", len(used_stations))
print("Zuordnung der LKWs zu den Ladesäulen:")
for new_idx, old_idx in enumerate(used_stations):
    lkw_list = solution[old_idx]
    lkw_str = ', '.join('LKW' + str(i + 1) for i in lkw_list)
    print(f"Ladesäule {new_idx + 1}: {lkw_str}")
"""
"""
import simpy

# Liste der LKW mit ihren genauen Ankunfts- und Abfahrtszeiten (in Stunden seit dem Start der Simulation)
TRUCKS = [
    {'ankunft': 0.5, 'abfahrt': 2.5},
    {'ankunft': 1.0, 'abfahrt': 4.0},
    {'ankunft': 1.5, 'abfahrt': 3.0},
    # Weitere LKW hier hinzufügen...
]

class ChargingStation:
    def __init__(self, env, num_stations):
        self.env = env
        self.station = simpy.Resource(env, capacity=num_stations)

    def charge(self, truck):
        charge_time = truck['abfahrt'] - truck['ankunft']
        yield self.env.timeout(charge_time)

def truck_charging(env, station, truck):
    yield env.timeout(truck['ankunft'])
    with station.station.request() as req:
        yield req
        yield env.process(station.charge(truck))
        print(f"LKW geladen: Ankunft um {truck['ankunft']}, Abfahrt um {truck['abfahrt']}")

# Start der Simulation
env = simpy.Environment()
station = ChargingStation(env, num_stations=1)  # Start mit einer Ladestation

for truck in TRUCKS:
    env.process(truck_charging(env, station, truck))

env.run()

# Ergebnis: Optimale Anzahl von Ladestationen berechnen
charged_trucks = len(station.station.queue) + len(station.station.users)
total_trucks = len(TRUCKS)
print(f"Anzahl der Ladestationen benötigt: {charged_trucks}/{total_trucks} LKW geladen.")
"""
"""
import random

def min_ladesaeulen(lkw_data):
    # LKWs nach Ankunftszeiten sortieren
    lkw_data.sort(key=lambda x: x[1])
    n = len(lkw_data)
    min_needed = int(0.8 * n)

    ladesaeulen = []
    lkw_to_ladesaeule = {}
    ladesaeule_count = 0
    geladen_count = 0

    for truck_id, ankunft, abfahrt in lkw_data:
        # Entferne alle LKWs aus den Ladesäulen, die bereits abgefahren sind
        ladesaeulen = [(endzeit, ladesaeule_id) for endzeit, ladesaeule_id in ladesaeulen if endzeit > ankunft]

        # Wenn es noch freie Ladesäulen gibt, benutze die
        if ladesaeulen:
            ladesaeule_id = ladesaeulen[0][1]
        else:
            # Neue Ladesäule hinzufügen
            ladesaeule_id = ladesaeule_count
            ladesaeule_count += 1

        ladesaeulen.append((abfahrt, ladesaeule_id))

        # Speichere die Zuordnung des LKW zur Ladesäule
        lkw_to_ladesaeule[truck_id] = ladesaeule_id

        # Zähle den geladenen LKW
        geladen_count += 1

        # Wenn wir mindestens 80% der LKW geladen haben, brechen wir ab
        if geladen_count >= min_needed:
            break

    # Ausgabe der Zuordnungen
    print(f"Minimale Anzahl an Ladesäulen: {ladesaeule_count}")
    print("LKW-Zuordnungen zu Ladesäulen:")
    for truck_id, ladesaeule_id in lkw_to_ladesaeule.items():
        print(f"LKW {truck_id} -> Ladesäule {ladesaeule_id}")

# Generiere Beispiel-LKW-Daten mit 1000 LKWs
def generate_lkw_data(num_lkw, max_time=1000, min_duration=1, max_duration=10):
    lkw_data = []
    for truck_id in range(1, num_lkw + 1):
        ankunft = random.randint(0, max_time)
        duration = random.randint(min_duration, max_duration)
        abfahrt = ankunft + duration
        lkw_data.append((truck_id, ankunft, abfahrt))
    return lkw_data

# Beispielwerte für 10 LKWs
random.seed(42)  # Für reproduzierbare Ergebnisse
lkw_data = generate_lkw_data(10)
print(lkw_data)
min_ladesaeulen(lkw_data)
"""
"""
# Liste der Ereignisse
events = []

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

# Ausgabe der maximal benötigten Ladesäulen
print(f"Wie viele Ladesäulen brauche ich? {max_stations_needed}")

# Ausgabe der LKW-Ladesäulen-Zuordnung
print("LKW-Ladesäulen-Zuordnung (bis 80% der LKWs):")
for lkw_id, station in lkw_to_station.items():
    print(f"LKW-ID {lkw_id} -> Ladesäule {station}")
"""
"""
import pandas as pd



anzahl_spalten = df.shape[1]

liste_cluster = []
for i in range(0, anzahl_spalten):
    liste_cluster.append('Cluster'+ str(i))
liste_ladesäulen = ['NCS', 'HPC', 'MCS']
# Dictionary zum Speichern der Ergebnisse
result = {}

# Über die Spalten iterieren

for l in range(0, anzahl_spalten):
    cluster_name ='Cluster' + str(l)
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
#########################################################################################

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

with open('Belegungspläne_Array/Cluster0_HPC.pkl', 'rb') as f:
    lkw_to_stations = pickle.load(f)
# Ausgabe der LKW-Ladesäulen-Zuordnung
print("LKW-Ladesäulen-Zuordnung (bis 80% der LKWs):")
for lkw_id, station in lkw_to_stations.items():
    print(f"LKW-ID {lkw_id} -> Ladesäule {station}")

# Ausgabe der maximal benötigten Ladesäulen
print(f"Wie viele Ladesäulen brauche ich? {max_stations_needed}")
"""
"""
def check_and_remove(array, a, b):
    # Überprüfen, ob alle Zahlen zwischen a und b im Array vorhanden sind
    if all(x in array for x in range(a, b + 1)):
        # Falls ja, entfernen wir diese Zahlen aus dem Array
        array = [x for x in array if x < a or x > b]
        return array
    else:
        # Falls nicht, geben wir den ursprünglichen Array zurück
        return array



data = [(1904, 10, 20),
        (1999, 15, 20),
        (1968, 20, 30),
        (1967, 0, 15),
        (1965, 5, 25),
        (1940, 15, 25),
        (1941, 25, 45),
        (1997, 5, 15),
        (1956, 20, 30),
        (1978, 10, 40)]

ladesäule= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]
zuordnung=[]
dict = {'Ladesäule1' : ladesäule}
i=1

print(dict)
for lkw in data:
    for key, value in dict.items():
        a = lkw[1]
        b = lkw[2]
        dummy = 0
        if all(x in value for x in range(a, b + 1)):
            # Falls ja, entfernen wir diese Zahlen aus dem Array
            array = [x for x in value if x < a or x > b]
            dict.update({key:array})
            zuordnung.append([lkw[0], key])
            break
        else:
            i+=1
            array = [x for x in ladesäule if x < a or x > b]
            dict['Ladesäule'+str(i)] = array
            zuordnung.append([lkw[0], 'Ladesäule'+str(i)])
        dummY = 0



print(zuordnung)
"""


from pulp import LpMinimize, LpProblem, LpVariable, lpSum

# Beispiel LKW-Liste
trucks = [
    (1, 4, 10, 'HPC'),  # (Ankunftszeit, Abfahrtszeit, benötigte Leistung)
    (2, 6, 10, 'NCS'),
]
dummy = lastgang_optimieren(trucks)



fasd = 0
