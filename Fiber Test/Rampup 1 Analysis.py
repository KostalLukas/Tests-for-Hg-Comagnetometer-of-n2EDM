# -*- coding: utf-8 -*-
"""
Fiber Power Rampup 1 Analysis v1.0

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


# function to discard np.nan elements from an array
def no_nan(arr):
    arr = arr[~np.isnan(arr)]
    return arr

# specify beamsplitter ratio R = reflected / transmitted
Rbs = 0.51

# relative error in the powermeter measurement
err = 0.05

# colors for plotting
colr = ['royalblue', 'limegreen', 'orange', 'red']

# load the data
# Fiber_080809_Ru_1.csv dataset contains DAQ voltages and powermeter measurement
# in uW for every 4 DAQ measurements
Vmod, V1, V2, P1, P2 = np.loadtxt('Data/Fiber_080809_Ru_1.csv', unpack=True, delimiter=',', skiprows=1)

# array of calibration constants for the photodiodes
cal1_arr = P1 / V1
cal2_arr = P2 / V2

# discard np.nan for when there is no powermeter measurement
Vpmod = Vmod[~np.isnan(P1)]
P1 = no_nan(P1)
P2 = no_nan(P2)
cal1_arr = no_nan(cal1_arr)
cal2_arr = no_nan(cal2_arr)

# absolute error in the powermeter measurement
P1_err = P1 * err
P2_err = P2 * err

# find mean calibration cosntant for the two channels
cal1_avg = np.mean(no_nan(cal1_arr))
cal2_avg = np.mean(no_nan(cal2_arr))

# calcualte uncertainty and peak to peak in calibration
cal1_err = np.sqrt(np.sum(P1_err**2)) / len(P1_err)
cal2_err = np.sqrt(np.sum(P2_err**2)) / len(P2_err)
cal1_ptp = np.ptp(cal1_arr)
cal2_ptp = np.ptp(cal2_arr)

# convert DAQ voltage to power in uW with uncertainty
Pv1 = cal1_avg * V1
Pv2 = cal2_avg * V2

Pv1_err = cal1_err / cal1_avg * Pv1
Pv2_err = cal2_err / cal2_avg * Pv2

# transmission from DAQ and power meter data also account for beamsplitter
Tv = Pv2 / Pv1 / Rbs
Tp = P2 / P1 / Rbs

# calculate expected uncertainty in transmission
Tv_err = Tv * np.sqrt((Pv2_err / Pv2)**2 + (Pv1_err / Pv1)**2)
Tp_err = Tp * np.sqrt((P2_err / P2)**2 + (P1_err / P1)**2)

# calcualte mean transmission and expected uncertainty in mean transmission
Tv_avg = np.mean(Tv)
Tp_avg = np.mean(Tp)

Tv_avgerr = np.sqrt(np.sum(Tv_err**2)) / len(Tv_err)
Tp_avgerr = np.sqrt(np.sum(Tp_err**2)) / len(Tp_err)

# calcualte peak to peak variation in transmission
Tv_ptp = np.ptp(Tv)
Tp_ptp = np.ptp(Tp)

# print the numerical results
file = 'Output/Fiber_Ru_1_results.txt'
open(file, 'w')

tprint('DAQ measurement:')
tprint(f'T_avg = {Tv_avg:.4g} uW')
tprint(f'T_err = {Tv_avgerr:.4g} uW')
tprint(f'T_ptp = {Tv_ptp:.4g} uW')
tprint()
tprint('Powermeter measurement:')
tprint(f'T_avg = {Tp_avg:.4g} uW')
tprint(f'T_err = {Tp_avgerr:.4g} uW')
tprint(f'T_ptp = {Tp_ptp:.4g} uW')
tprint()
tprint('Calibration constants:')
tprint(f'Ch1_avg = {cal1_avg:.4g} uW a.u.^-1')
tprint(f'Ch1_err = {cal1_err:.4g} uW a.u.^-1')
tprint(f'Ch1_ptp = {cal1_ptp:.4g} uW a.u.^-1')
tprint(f'Ch2_avg = {cal1_avg:.4g} uW a.u.^-1')
tprint(f'Ch2_err = {cal1_err:.4g} uW a.u.^-1')
tprint(f'Ch2_ptp = {cal1_ptp:.4g} uW a.u.^-1')

# parameters for plotting input and output power
plt.figure(1)
plt.title('Fiber Input and Output Power against Modulation Voltage', pad=40)
plt.xlabel('modulation voltage $V_{mod}$ (V)')
plt.ylabel('power $P$ ($\mu W$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(Vmod, Pv1, ls='-', c=colr[0], label='$P_{fi}$ DAQ')
plt.plot(Vpmod, P1, ls='-', c=colr[3], label='$P_{fi}$ powermeter')

plt.plot(Vmod, Pv2, ls='--', c=colr[0], label='$P_{fo}$ DAQ')
plt.plot(Vpmod, P2, ls='--', c=colr[3], label='$P_{fo}$ powermeter')

plt.fill_between(Vmod, Pv1-Pv1_err, Pv1+Pv1_err, color=colr[0], alpha=0.2)
plt.fill_between(Vpmod, P1-P1_err, P1+P1_err, color=colr[3], alpha=0.2)
plt.fill_between(Vmod, Pv2-Pv2_err, Pv2+Pv2_err, color=colr[0], alpha=0.2)
plt.fill_between(Vpmod, P2-P2_err, P2+P2_err, color=colr[3], alpha=0.2)

plt.legend(loc=(-0.2, 1.05), ncol=4)

# save plot
plt.savefig('Output/Fiber_Ru_2_power.png', dpi=300, bbox_inches='tight')

# parameters for plotting transmission against power
plt.figure(2)
plt.title('Fiber Transmission against Input Power', pad=40)
plt.xlabel('input power $P_{fi}$ ($\mu W$)')
plt.ylabel('transmission $T = P_{fo} / P_{fi}$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.errorbar(Pv1, Tv, xerr=Pv1_err, yerr=Tv_err, fmt='x', capsize=4, \
             c=colr[0], label='DAQ')
plt.errorbar(P1, Tp, xerr=P1_err, yerr=Tp_err, fmt='x', capsize=4, \
             c=colr[3], label='Thorlabs powermeter')

plt.legend(loc=(0, 1.05), ncol=2)

# save plot
plt.savefig('Output/Fiber_Ru_1_transmission.png', dpi=300, bbox_inches='tight')

# parameters for plotting calibration constants against power
plt.figure(3)
plt.title('DAQ Calibration Constants against Incident Power', pad=40)
plt.xlabel('incident power $P$ ($\mu W$)')
plt.ylabel('calibration $c$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.errorbar(P1, cal1_arr, xerr=P1_err, yerr=cal1_err, fmt='x', capsize=4, \
             c=colr[1], label='Ch1 calibration')
plt.errorbar(P2, cal2_arr, xerr=P1_err, yerr=cal1_err, fmt='x', capsize=4, \
             c=colr[2], label='Ch2 calibration')

plt.legend(loc=(0, 1.05), ncol=2)

# save plot
plt.savefig('Output/Fiber_Ru_1_calibration.png', dpi=300, bbox_inches='tight')

# show the plots
plt.show()
