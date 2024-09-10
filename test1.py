import random


def berechne_akkustand(row_idx):
    if 0 <= row_idx <= 359:
        # Akkustand nachts: 20% +/- 10%
        grund_akkustand = 0.20
        schwankung = random.uniform(-0.10, 0.10)
        akkustand = grund_akkustand + schwankung
    elif 360 <= row_idx <= 1440:
        # Linearer Verlauf zwischen 60% (6:00) und 20% (24:00)
        start_akkustand = 0.60
        end_akkustand = 0.20
        tages_minuten = 1440 - 360  # Anzahl der Minuten zwischen 6:00 und 24:00
        akkustand = start_akkustand - ((start_akkustand - end_akkustand) * (row_idx - 360) / tages_minuten)
        schwankung = random.uniform(-0.10, 0.10)
        akkustand += schwankung
    else:
        return "Ungültiger row_idx, muss zwischen 0 und 1440 liegen."

    # Akkustand begrenzen, damit er nicht unter 0% oder über 100% liegt
    akkustand = max(0, min(1, akkustand))

    return round(akkustand, 2)


# Beispielaufruf
print(berechne_akkustand(180))  # Ein Wert zwischen 0 und 359 (Nacht)
print(berechne_akkustand(720))  # Ein Wert zwischen 360 und 1440 (Tag)
