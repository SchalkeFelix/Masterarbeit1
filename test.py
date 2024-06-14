import gurobipy as gp
from gurobipy import GRB
from Initialiserung import*
from methods import*
import ast
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

