# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gurobipy as gp
from approximation import *
from plots import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x, verkehrsdaten = read_lkw_data(csv_dateipfad)
    plot_verkehr(verkehrsdaten)