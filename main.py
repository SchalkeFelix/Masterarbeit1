# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from scipy.optimize import linprog



# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # Koeffizienten der Zielfunktion (Anzahl der Kabel minimieren)
    c = [200, 100, 1000, 1000]  # Wir minimieren x1 + x2

    # Ungleichungsbedingungen (Leistungsgleichung: 100*x1 + 40*x2 >= 250)
    A = [[-100, -50, -20, -10]]  # Koeffizienten der Ungleichung (linke Seite der Gleichung)
    b = [-250]  # Rechte Seite der Ungleichung

    # Bereich für x1 und x2 (ganzzahlige Werte)
    bounds = [(0, None), (0, None),(0, None), (0, None)]  # x1, x2 >= 0

    # Optimierung durchführen
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    # Ergebnisse ausgeben
    x1 = int(round(result.x[0]))  # Anzahl der 100 kW-Kabel
    x2 = int(round(result.x[1]))  # Anzahl der 40 kW-Kabel
    x3 = int(round(result.x[2]))  # Anzahl der 100 kW-Kabel
    x4 = int(round(result.x[3]))  # Anzahl der 40 kW-Kabel

    print("Optimale Anzahl der 100 kW-Kabel:", x1)
    print("Optimale Anzahl der 50 kW-Kabel:", x2)
    print("Optimale Anzahl der 20 kW-Kabel:", x3)
    print("Optimale Anzahl der 10 kW-Kabel:", x4)
    print("Gesamtanzahl der Kabel:", x1 + x2+ x3+x4)
