import pandas as pd
import numpy as np
import datetime
from methods import *
from datetime import date, timedelta
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt

# Beispiel DataFrame
data = {
    'Jahr': [2010, 2011, 2012, 2013, 2014],
    'Umsatz': [50000, 60000, 75000, 90000, 100000],
    'Gewinn': [2000, 4000, 6000, 8000, 10000]
}

df = pd.DataFrame(data)
df.set_index('Jahr', inplace=True)  # Setze das Jahr als Index
print(df)

# Liniendiagramm erstellen
plt.figure(figsize=(10, 5))  # Größe des Diagramms festlegen

# Plotten der Linien für 'Umsatz' und 'Gewinn'
plt.plot(df.index, df['Umsatz'], marker='o', label='Umsatz', color='blue', linestyle='solid')
plt.plot(df.index, df['Gewinn'], marker='o', label='Gewinn', color='blue', linestyle='solid')

plt.show()
