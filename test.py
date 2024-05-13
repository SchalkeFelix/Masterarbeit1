import datetime

jahr = 2021
start_datum = 210104  # YYMMDD für den 4. Januar 2021

montage_liste = []
sonntage_liste =[]

# Schleife für jeden Monat im Jahr
for monat in range(1, 13):  # von 1 bis 12 (einschließlich 12)
    tage_im_monat = 31  # wir setzen angenommen erst einmal auf 31 tage
    # je nach dem monat
    if monat == 4 or monat == 6 or monat == 9 or monat == 11:
        tage_im_monat = 30
    elif monat == 2:
        # februar hat verschiedene tage
        if (jahr % 4 == 0 and jahr % 100 != 0) or (jahr % 400 == 0):
            tage_im_monat = 29
        else:
            tage_im_monat = 28
    # Schleife für jeden Tag im aktuellen Monat
    for tag in range(1, tage_im_monat + 1):
        # Formatierung des Datums im gewünschten Stil YYMMDD
        datum_str = f"{str(jahr)[-2:]}{str(monat).zfill(2)}{str(tag).zfill(2)}"
        datum_int = int(datum_str)
        if datum_int >= start_datum:
            # Prüfen, ob der Tag ein Montag ist (Wochentagnummer 0 für Montag)
            if datetime.date(jahr, monat, tag).weekday() == 0:
                montage_liste.append(datum_int)
            if datetime.date(jahr, monat, tag).weekday() == 6:
                sonntage_liste.append(datum_int)

print(montage_liste)
print(sonntage_liste)

if len(montage_liste) != len(sonntage_liste):
    montage_liste.pop()

print(montage_liste)
print(sonntage_liste)