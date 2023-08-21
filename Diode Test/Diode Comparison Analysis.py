# -*- coding: utf-8 -*-
"""
Diode Comparison Analysis v1.0

Lukas Kostal, 18.8.2023, PSI
"""


import numpy as np
from matplotlib import pyplot as plt


# function to print to console and file simultaneously
def tprint(text=''):
    print(text)
    global file
    with open(file, 'a') as output:
        print(text, file=output)


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
        V = V[0: 105]
        n = 105

    # array of sampling times
    t = np.arange(n) / fs

    # calculate mean, standard deviation and standard error of measured voltage
    V_avg[i] = np.mean(V)
    V_std[i] = np.std(V)
    V_sem[i] = V_std[i] / np.sqrt(len(V))

    # check if should be plotted and plot
    if pds[i] in plot:
        plt.plot(t, V, label=pds[i])

if len(plot) > 0:
    plt.legend(loc=(-0.2, 1.05), ncol=len(plot))
    plt.savefig('Output/Comparison_voltages.png', dpi=300, bbox_inches='tight')

P_sem = V_sem * P / V_avg

# print the numerical results
file = 'Output/Comparison_results.csv'
open(file, 'w')

tprint('diode, mean voltage (V), standard deviation (V), standard error (V)')

for i in range(0, len(pds)):
    tprint(f'{pds[i]}, {V_avg[i]:.6g}, {V_std[i]:.4g}, {V_sem[i]:.4g}')

# parameters for plotting mean measured voltage for each photodiode
plt.figure(2)
plt.title('Mean DAQ Input Voltage')
plt.xlabel('photodiode')
plt.ylabel('voltage $V$ (V)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.errorbar(pds, V_avg, yerr=V_sem, marker='x', ls='none', capsize=5, c=colr[0])
plt.savefig('Output/Comparison_mean.png', dpi=300, bbox_inches='tight')

# parameters for plotting SEM of power for each photodiode
plt.figure(3)
plt.title('SEM of Calculated Power')
plt.xlabel('photodiode')
plt.ylabel('SEM $\sigma_{P}$ ($\mu W$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(pds, P_sem, marker='x', ls='none', c=colr[0])
plt.savefig('Output/Comparison_sem.png', dpi=300, bbox_inches='tight')
plt.show()

# show the plots
plt.show()
