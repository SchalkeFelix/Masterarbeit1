import pandas as pd
import numpy as np
import datetime
from methods import *
from datetime import date, timedelta
import pandas as pd
import os

# Ordnerpfad, in dem die CSV-Dateien gespeichert werden sollen
output_folder = 'output_dataframes'

# Erstelle den Ordner, falls er nicht existiert
os.makedirs(output_folder, exist_ok=True)

for i in range(5):  # Beispiel-Schleife, um 5 DataFrames zu erstellen
        # DataFrame erstellen (hier als Beispiel mit Dummy-Daten)
        data = {'Column1': [i, i + 1, i + 2],
                'Column2': ['A', 'B', 'C']}

        df = pd.DataFrame(data)

        # Dateipfad für die CSV-Datei
        csv_filename = os.path.join(output_folder, f'dataframe_{i}.csv')

        # DataFrame als CSV-Datei speichern
        df.to_csv(csv_filename, index=False)