import pandas as pd
import numpy as np
import datetime

"""
# Beispiel DataFrame erstellen
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Helen'],
        'Age': [25, 30, 35, 40, 45, 50, 55, 60],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego']}
df = pd.DataFrame(data)

# Anzahl der gewünschten Teile
num_parts = 3

# DataFrame in mehrere Teile aufteilen basierend auf der Anzahl der Zeilen
dfs = np.array_split(df, num_parts)

# Ergebnisse anzeigen
print(df)
for i, part_df in enumerate(dfs):
    print(f"Teil DataFrame {i + 1}:")
    print(part_df)
    print()
"""

"""
# Beispiel DataFrame erstellen
data = {'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e']}
df = pd.DataFrame(data)

# Anzahl der Zeilen, die gelöscht werden sollen
n = 3

# Die ersten n Zeilen löschen
df = df.drop(index=range(n))

# Ergebnis anzeigen
print(df)
"""

"""
import pandas as pd

# Beispiel DataFrame erstellen
data = {'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e']}
df = pd.DataFrame(data)

# Anzahl der Zeilen, die gelöscht werden sollen
n = 2  # Anzahl der letzten Zeilen, die gelöscht werden sollen

# Die letzten n Zeilen löschen
df = df.drop(index=df.tail(n).index)

# Ergebnis anzeigen
print(df)
"""




def first_monday_of_year(year):
    # Erstes Datum des Jahres
    first_day_of_year = datetime.date(year, 1, 1)

    # Tag der Woche für das erste Datum des Jahres (0 = Montag, 1 = Dienstag, ..., 6 = Sonntag)
    first_day_of_week = first_day_of_year.weekday()

    # Wenn der erste Tag des Jahres ein Montag ist (0), ist er der erste Montag des Jahres
    if first_day_of_week == 0:
        return 1  # Der erste Montag ist am 1. Tag des Jahres

    # Ansonsten finden wir den Tag des ersten Montags des Jahres
    # Berechnen, wie viele Tage bis zum nächsten Montag verbleiben
    days_until_next_monday = (7 - first_day_of_week) % 7

    # Tag des Jahres, an dem der erste Montag liegt
    first_monday_day_of_year = days_until_next_monday + 1

    return first_monday_day_of_year

# Beispielaufruf
year = 2021
first_monday_day = first_monday_of_year(year)
print(f"Der erste Montag im Jahr {year} ist der {first_monday_day}. Tag des Jahres.")



def days_until_end_of_year(year):
    # Letztes Datum des Jahres
    last_day_of_year = datetime.date(year, 12, 31)

    # Tag der Woche für das letzte Datum des Jahres (0 = Montag, 1 = Dienstag, ..., 6 = Sonntag)
    last_day_of_week = last_day_of_year.weekday()

    # Tag des Jahres, an dem der letzte Sonntag liegt
    last_sunday = (last_day_of_year - datetime.timedelta(days=last_day_of_week + 1)).timetuple().tm_yday

    # Anzahl der Tage bis zum Jahresende (365 Tage für normale Jahre und 366 Tage für Schaltjahre)
    days_until_end = 365 if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0) else 366
    days_until_end -= last_sunday  # Anzahl der Tage vom letzten Sonntag bis zum Jahresende

    return days_until_end

# Beispielaufruf für das Jahr 2021
year = 2024

days_until_end = days_until_end_of_year(year)
print(f"Anzahl der Tage zwischen dem letzten Sonntag und dem Ende des Jahres {year}: {days_until_end} Tage.")
