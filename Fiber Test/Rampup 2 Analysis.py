# -*- coding: utf-8 -*-
"""
Fiber Power Rampup 2 Analysis v1.0

Lukas Kostal, 8.8.2023, PSI
"""

import numpy as np
from matplotlib import pyplot as plt


# function to print to console and file simultaneously
def tprint(text=''):
    print(text)
    global file
    with open(file, 'a') as output:
        print(text, file=output)


# relative error in the powermeter measurement
err = 0.05

# colors for plotting
colr = ['blue', 'green', 'orange', 'red']

# load the data
# Fiber_080817_Ru_2.csv dataset cotains data from two power rampups
Vmod, Pfi1, Pfo1, Pfi2, Pfo2 = np.loadtxt('Data/Fiber_080817_Ru_2.csv', unpack=True, delimiter=',', skiprows=1)

# calcualte absolute errors in power
Pfi1_err = Pfi1 * err
Pfo1_err = Pfo1 * err
Pfi2_err = Pfi2 * err
Pfo2_err = Pfo2 * err

# calcualte fiber transmission for 1st ramp
T1 = Pfo1 / Pfi1
T1_err = T1 * np.sqrt(2 * err**2)

# calcualte fiber transmission for 2nd ramp
T2 = Pfo2 / Pfi2
T2_err = T2 * np.sqrt(2 * err**2)

# get mean, standard deviation and standard error on the mean of transmission
avg1 = np.mean(T1)
avg2 = np.mean(T2)
std1 = np.std(T1)
std2 = np.std(T2)
sem1 = std1 / np.sqrt(len(T1))
sem2 = std2 / np.sqrt(len(T2))

# get peak to peak variation in transmission
ptp1 = np.ptp(T1)
ptp2 = np.ptp(T2)

# print the numerical results
file = 'Output/Fiber_Ru_2_results.txt'
open(file, 'w')

tprint('Rampup 1:')
tprint(f'T_avg = {avg1:.4g} uW')
tprint(f'T_sem = {sem1:.4g} uW')
tprint(f'T_std = {std1:.4g} uW')
tprint(f'T_ptp = {ptp1:.4g} uW')
tprint()
tprint('Rampup 2:')
tprint(f'T_avg = {avg2:.4g} uW')
tprint(f'T_sem = {sem2:.4g} uW')
tprint(f'T_std = {std2:.4g} uW')
tprint(f'T_ptp = {ptp2:.4g} uW')

# parameters for plotting input and output power
plt.figure(1)
plt.title('Fiber Input and Output Power against Modulation Voltage', pad=40)
plt.xlabel('modulation voltage $V_{mod}$ (V)')
plt.ylabel('power $P$ ($\mu W$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(Vmod, Pfi1, ls='-', c=colr[0], label='$P_{fi}$ rampup 1')
plt.plot(Vmod, Pfi2, ls='-', c=colr[3], label='$P_{fi}$ rampup 2')
plt.plot(Vmod, Pfo1, ls='--', c=colr[0], label='$P_{fi}$ rampup 1')
plt.plot(Vmod, Pfo2, ls='--', c=colr[3], label='$P_{fo}$ rampup 2')

plt.fill_between(Vmod, Pfi1-Pfi1_err, Pfi1+Pfi1_err, color=colr[0], alpha=0.2)
plt.fill_between(Vmod, Pfi2-Pfi2_err, Pfi2+Pfi2_err, color=colr[3], alpha=0.2)
plt.fill_between(Vmod, Pfo1-Pfo1_err, Pfo1+Pfo1_err, color=colr[0], alpha=0.2)
plt.fill_between(Vmod, Pfo2-Pfo2_err, Pfo2+Pfo2_err, color=colr[3], alpha=0.2)

plt.legend(loc=(-0.2, 1.05), ncol=4)

# save plot
plt.savefig('Output/Fiber_Ru_2_power.png', dpi=300, bbox_inches='tight')

# parameters for plotting transmission
plt.figure(2)
plt.title('Fiber Transmission against Input Power', pad=40)
plt.xlabel('input power $P_{fi}$ ($\mu W$)')

plt.ylabel('transmission $T = P_{fo} / P_{fi}$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(Pfi1, T1, c=colr[0], label='rampup 1')
plt.plot(Pfi2, T2, c=colr[3], label='rampup 2')

plt.fill_between(Pfi1, T1-T1_err, T1+T1_err, color=colr[0], alpha=0.2)
plt.fill_between(Pfi2, T2-T2_err, T2+T2_err, color=colr[3], alpha=0.2)

plt.legend(loc=(0, 1.05), ncol=2)

# save plot
plt.savefig('Output/Fiber_Ru_2_transmission.png', dpi=300, bbox_inches='tight')

# show the plots
plt.show()
