import pandas as pd
import numpy as np
import datetime
from methods import *
from datetime import date, timedelta
import pandas as pd
import os



my_dict = {
    0: [13, 14, 19, 29, 30, 31, 32, 33, 34, 35, 44],
    1: [1, 51],
    2: [2, 3, 4, 6, 7, 21, 22, 28, 36, 37, 38, 39, 40, 41, 42, 45],
    3: [5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 20, 23, 24, 25, 26, 27, 43, 46, 47, 48, 49, 50]
}

# Durchlauf des Dictionaries in aufsteigender Reihenfolge der Schlüssel
for key in sorted(my_dict.keys()):
    # Zugriff auf die Liste für den aktuellen Schlüssel und Durchlaufen der Liste
    current_list = my_dict[key]
    print(f"Liste mit Index {key}:")
    for item in current_list:
        print(type(item))
        print(item)