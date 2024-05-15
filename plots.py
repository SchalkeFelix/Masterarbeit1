import pandas as pd
import datetime
import matplotlib
matplotlib.use('TkAgg')  # Oder ein anderes unterstütztes Backend
import matplotlib.pyplot as plt
import os
import numpy as np
from config import *

show_plots = True
save_plots = True
folder = 'PLOTS'

def plot_verkehr(verkehrsdaten, Wochen_name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    x_values = verkehrsdaten['x_values']
    y_values = verkehrsdaten['y_values']
    x_continuous = verkehrsdaten['x_continuous']
    y_continuous = verkehrsdaten['y_continuous']
    differenzen = verkehrsdaten['differenzen']

    fig, ax = plt.subplots()
    # Plot der Original-Stufenfunktion und der approximierten stetigen Funktion
    plt.step(x_values, y_values, label='Stufenfunktion', where='post')
    plt.plot(x_continuous, y_continuous, label='Approximierte Funktion')
    plt.step(x_continuous, y_continuous, label='Approximierte Stufenfunktion', where='post')

    # Markieren der Stunden wo sich die Ausgangsdaten und die approximierten Daten um mehr als 20 LKWs unterscheiden
    # for i, diff in enumerate(differenzen):
    #     if diff > 20:
    #         plt.scatter(x_values[i], y_values[i], color='red')
    #plt.xlim(0,24)
    #plt.ylim(0,720)
    plt.xlabel('Zeit [h]')
    plt.ylabel('Anzahl der LKWs pro timestep')
    #plt.legend()

    file_name = f"{Wochen_name}_verkehr.png"
    file_path = os.path.join(folder, file_name)
    plt.savefig(file_path)

    if show_plots:
        plt.show()