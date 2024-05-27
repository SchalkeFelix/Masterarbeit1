import pandas as pd
import numpy as np
import random

'''
def generate_entry(df_name):
    # Eintrag 1: Abhängig vom Namen des DataFrames
    if df_name == 'DF1':
        entry1 = 'HPC'
    elif df_name == 'DF2':
        entry1 = 'MCS'
    else:
        entry1 = 'NCS'

    # Eintrag 2: Eine zufällige Zahl zwischen 0,05 und 0,3
    entry2 = round(random.uniform(0.05, 0.3), 3)

    # Eintrag 3: Eine Zahl entweder 252, 504 oder 756 mit Wahrscheinlichkeiten 0,4, 0,2, 0,4
    entry3 = random.choices([252, 504, 756], [0.4, 0.2, 0.4])[0]

    return [entry1, entry2, entry3]


def process_dataframes(dataframes):
    # Initialisieren des resultierenden DataFrames mit der gleichen Struktur wie die Eingabedaten
    result_dict = {col: [] for col in dataframes[0].columns}

    # Durchlaufen der Zeilen
    num_rows = max(df.shape[0] for df in dataframes)
    for row_idx in range(num_rows):
        row_data = {col: [] for col in dataframes[0].columns}
        for df in dataframes:
            if row_idx < df.shape[0]:
                for col in df.columns:
                    value = df.at[row_idx, col]
                    arrays = [generate_entry(df.name) for _ in range(value)]
                    row_data[col].extend(arrays)
        for col in row_data:
            result_dict[col].append(row_data[col])

    # Erstellen eines DataFrames aus dem Dictionary der Arrays
    result_df = pd.DataFrame(result_dict)
    return result_df


'''
# Beispiel-Datenframes
df1 = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
})
df1.name = 'DF1'

df2 = pd.DataFrame({
    'A': [2, 3],
    'B': [1, 5]
})
df2.name = 'DF2'

# Liste der DataFrames
dataframes = [df1, df2]

x = df1.name
print(x)
print(type(x))

