import numpy as np
from methods import*
import gurobipy as gp
from gurobipy import GRB

############################################ Initialisierung Optimierung Ladeanschluss ################################
r = 150  # Anzahl der Transformator-Klassen
m = 50  # Anzahl der Ladesäulen

errechnete_Ladeleistung = {'NCS': 88, 'HPC': 350, 'MCS': 1000}   # abhängig, ob Lademanagement oder nicht

P = [50, 100, 160, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150] * 10 # Leistungskapazitäten der Transformatoren (für jede Klasse)

l = ladeleistung_array(8, 39, 3, errechnete_Ladeleistung)  # HPC, NCS, MCS

l_installiert = ladeleistung_array(8, 39, 3, ladeleistung_liste)
kipppunkt_hs_trafo = 20000

f = [18284.15, 23850.40, 28772.70, 40394.15, 46956.48, 55055.90, 63520.40, 74290.05, 92056.90, 107417.90, 131328.15,
     161348.90, 193641.90, 233614.40, 285106.65] * 10 # Festkosten für die Auswahl eines Transformators jeder Klasse

kabel_mittelspannung = [(35, 9.27),
                        (50, 9.55),
                        (70, 11.82),
                        (95, 14.89),
                        (120, 17.87),
                        (150, 17.76),
                        (185, 21.34),
                        (240, 26.94),
                        (300, 36.93),
                        (400, 46.93),
                        (500, 57.95),
                        (630, 86.89),
                        (800, 103.38)
                        ]  # (Querschnitt, Preis/m)
kabel_niederspannung = [(4, 0.59),
                        (6, 0.7),
                        (10, 1.01),
                        (16, 1.57),
                        (25, 2.47),
                        (35, 3.34),
                        (50, 4.48),
                        (70, 6.33),
                        (95, 8.61),
                        (120, 10.53),
                        (150, 13.22),
                        (185, 16.19),
                        (240, 20.81),
                        (300, 26.06),
                        (400, 35.04)
                        ]  # (Querschnitt, Preis/m)

################################################# Optimeriung Anzahl der Trafos ########################################

# Initialisiere das Modell
model = gp.Model("Minimize_Total_Costs_with_Transformers")

# Entscheidungsvariablen x_ij (Binary) und y_i (Binary)
x = model.addVars(r, m, vtype=GRB.BINARY, name="x")
y = model.addVars(r, vtype=GRB.BINARY, name="y")

# Zielfunktion: Minimiere die Kosten der ausgewählten Transformatoren
model.setObjective(gp.quicksum(f[i] * y[i] for i in range(r)), GRB.MINIMIZE)

# 1. Jede Ladesäule wird genau einem Transformator zugewiesen
for j in range(m):
    model.addConstr(gp.quicksum(x[i, j] for i in range(r)) == 1, name=f"ladesaeule_{j}")

# 2. Leistungskapazität der Transformatoren (nur wenn die Klasse ausgewählt wurde)
for i in range(r):
    model.addConstr(gp.quicksum(x[i, j] * l[j] for j in range(m)) <= P[i] * y[i], name=f"kapazitaet_{i}")

# Optimierung
model.optimize()

# Status der Lösung
print("Status:", model.Status)

kosten_transformator = model.objVal

# Zugewiesene Transformatoren
print("\nZugewiesene Transformatoren:")
for i in range(r):
    if y[i].X > 0.5:  # Wegen Rundungsfehlern bei binären Variablen > 0.5
        print(f"{int(y[i].X)} Transformator(en) der Klasse {i+1} (Kapazität {P[i]} kW) ausgewählt.")

# Zuweisungen der Ladesäulen zu Transformatoren
print("\nZuweisungen der Ladesäulen zu Transformatoren:")
transformator_ladesaeulen = {}

for i in range(r):
    zugeordnete_ladesaeulen = []
    for j in range(m):
        if x[i, j].X > 0.5:
            zugeordnete_ladesaeulen.append(j + 1)  # Ladesäule-Index um 1 erhöht, um menschlich verständlich zu sein
    if zugeordnete_ladesaeulen:
        transformator_ladesaeulen[i + 1] = zugeordnete_ladesaeulen  # Transformator-Index um 1 erhöht

# Ausgabe der Liste: Transformator und zugeordnete Ladesäulen
for transformator, ladesaeulen in transformator_ladesaeulen.items():
    # Erstelle eine neue Liste, die die Ladesäule und deren Leistung als String enthält
    ladesaeulen_mit_leistung = [f"{ladesaeule} ({l[ladesaeule - 1]} kW)" for ladesaeule in ladesaeulen]

    # Gebe die Liste aus
    print(f"Transformator {transformator}: Ladesäulen {ladesaeulen_mit_leistung}")
print(transformator_ladesaeulen)
print(kosten_transformator)
############################################ Optimierung Niederspannungskabel #########################################


data = transformator_ladesaeulen
Transformator_Leistung_gesamt = {}
kabel_kosten_niederspannung = 0
# Process each transformer
for transformer, ladesäulen in data.items():
    transformator_leistung = 0
    results = process_transformer(ladesäulen, l, True, kabel_niederspannung)
    print(f"Transformer {transformer}:")
    for ladesäule, details in results.items():
        cable_info = details['Selected Cable']
        if cable_info:
            cable_diameter, cable_price = cable_info
            transformator_leistung += details['Performance']
            kabel_kosten_niederspannung += (details['Cable Length']*cable_price)
            print(f"  Ladesäule {ladesäule}: Performance={details['Performance']}, Cable Length={details['Cable Length']}m, Cable Diameter={details['Cable Diameter']:.2f}mm, Selected Cable Diameter={cable_diameter}mm, Price={cable_price}€")
        else:
            print(f"  Ladesäule {ladesäule}: Performance={details['Performance']}, Cable Length={details['Cable Length']}m, Cable Diameter={details['Cable Diameter']:.2f}mm, No suitable cable found")
    Transformator_Leistung_gesamt[transformer]=transformator_leistung

print(kabel_kosten_niederspannung)
print(Transformator_Leistung_gesamt)

################################################ Optimierung Mittelspannungskabel ######################################
kabel_kosten_mittelspannung = 0
kabellaenge = 3000
for transformer in Transformator_Leistung_gesamt:
    leistung = Transformator_Leistung_gesamt[transformer]
    diameter_ms = assign_cable_diameter(leistung, kabellaenge, False)
    cable = select_cable(diameter_ms, kabel_mittelspannung)
    kabel_kosten_mittelspannung += (kabellaenge * cable[1])
    print('Transformator ' + str(transformer) + ' wird mit Kabel der Dicke ' + str(cable[0]) + ' mm^2 angeschlossen und kostet ' + str(cable[1]) + ' €/m !')
    dummy = 0

############################################## Anschluss an Netzanschlusspunkt #########################################

angeschlossene_leistung = sum(l_installiert)

if angeschlossene_leistung >= kipppunkt_hs_trafo:
    print('Es wird ein HS-Trafo benötigt!')
    kosten_hs_trafo = 500000
else:
    print('Es wird kein HS-Trafo benötigt!')
    kosten_hs_trafo = 0


############################################# Gesamt Kosten ###########################################################

gesamt_kosten = kosten_hs_trafo + 2* (kabel_kosten_mittelspannung + kabel_kosten_niederspannung) + kosten_transformator

print('Die Gesamtkosten betragen: ' + str(round(gesamt_kosten, 2)) + ' € !')