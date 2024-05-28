import pandas as pd

# Beispiel-Datenframe erstellen
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}

df = pd.DataFrame(data)
print(df)
# Indizes als Liste zurückgeben
index_list = df.index.tolist()

print(index_list)