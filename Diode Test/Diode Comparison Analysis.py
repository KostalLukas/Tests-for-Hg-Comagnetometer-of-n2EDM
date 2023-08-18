# -*- coding: utf-8 -*-
"""
Diode Comparison Analysis v1.0

Lukas Kostal, 18.8.2023, PSI
"""


import numpy as np
from matplotlib import pyplot as plt


# list of photodiodes to be tested
pds = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']

# list of photoidoes for which to plot measured power over time
plot = pds

# sampling frequency in Hz
fs = 10

# power of laser during test in uW
P = 1680

# empty arrays to hold results for each diode
V_avg = np.zeros(len(pds))
V_std = np.zeros(len(pds))
V_sem = np.zeros(len(pds))

# colors for plotting
colr = ['royalblue', 'limegreen', 'orange']

if len(plot) > 0:
    # parameters for plotting measured power over time
    plt.figure(1)
    plt.title('DAQ Input Voltage over Time', pad=40)
    plt.xlabel('time $t$ (s)')
    plt.ylabel('voltage $V$ (V)')
    plt.rc('grid', linestyle=':', c='black', alpha=0.8)
    plt.grid()

# loop over all specified photodiodes
for i in range(0, len (pds)):
    # load the data for the given photodiode
    V = np.genfromtxt(f'Data/Diode_{pds[i]}.txt', usecols=(0), unpack=True, delimiter=',')

    # set all -ve measurements equal to 0
    V[V <= 0] = 0

    # discrad fist and last 3 measurements and convert to V
    V = V[3:-3] * 10
    V[V <= 0] = 0

    # get no of measurements for given photodiode
    n = len(V)

    # if test lasted more than 150s crop it at 110s
    if n > 150:
        V = V[0: 110]
        n = 110

    # array of sampling times
    t = np.arange(n) / fs

    # calculate mean and standard deviation of the measured voltage
    V_avg[i] = np.mean(V)
    V_std[i] = np.std(V)

    # check if should be plotted and plot
    if pds[i] in plot:
        plt.plot(t, V, label=pds[i])

if len(plot) > 0:
    plt.legend(loc=(-0.2, 1.05), ncol=len(plot))
    plt.savefig('Output/Comparison_voltages.png', dpi=300, bbox_inches='tight')

# calculate standard error on the mean for measured votlage
V_sem = np.std(V)

# parameters for plotting mean measured voltage for each photodiode
plt.figure(2)
plt.title('Mean DAQ Input Voltage')
plt.xlabel('photodiode')
plt.ylabel('voltage $V$ (V)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.errorbar(pds, V_avg, yerr=V_sem, marker='x', ls='none', capsize=5, c=colr[0])
plt.savefig('Output/Comparison_mean.png', dpi=300, bbox_inches='tight')
plt.show()

# show the plots
plt.show()
