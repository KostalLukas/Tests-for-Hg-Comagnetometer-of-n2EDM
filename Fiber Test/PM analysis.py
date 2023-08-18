# -*- coding: utf-8 -*-
"""
Fiber Test Power Meter Analysis v2.0

Lukas Kostal, 26.7.2023, PSI
"""


import numpy as np
from matplotlib import pyplot as plt


# function to print to console and file simultaneously
def tprint(text=''):
    print(text)
    global file
    with open(file, 'a') as output:
        print(text, file=output)


# function to calcualte mean and standard error on the mean
def get_mean(arr):
    n = len(arr) - np.sum(np.isnan(arr))
    mean = np.nansum(arr) / n
    sem = np.sqrt(np.nansum( (arr - mean)**2 )) / n
    
    return(mean, sem)
    

# relative error in power meter measurements
err = 0.05

# specify colors for plotting
clr = ['blue', 'limegreen', 'orange']

# load the data
data = np.loadtxt('Data/Powermeter_data.csv', unpack=True, delimiter=',', skiprows=1)

# number of measurements
n = len(data[0, :])

# array to hold date in month, day, hour, minute format
date = data[0:4, :]

# specify no of days in each month
# taking 2024 to be a leap year otherwise change 29 -> 28
mtd = np.array([31, 30, 29, 31, 30, 31, 30, 31, 30, 31, 30, 31])

# convert date into elapsed time in h
t = np.zeros(n)
for i in range(0, n):
    t[i] += (date[0, i] - date[0, 0]) * mtd[int(date[0, i] - 1)] * 24
    t[i] += (date[1, i] - date[1, 0]) * 24    
    t[i] += (date[2, i] - date[2, 0])
    t[i] += (date[3, i] - date[3, 0]) / 60

# convert power to mW
data[4:9, :] /= 1000

# slice the data into measurements P in uW and ADC in V
Ptot = data[4, :] 
Pbst = data[5, :]
Pbsr = data[6, :]
Pfi  = data[7, :]
Pfo  = data[8, :]
Ch1  = data[9, :]
Ch2  = data[10, :]

# calcualte absolute error in uW
Ptot_err = Ptot * err
Pbsr_err = Pbst * err
Pbsr_err = Pbsr * err
Pfi_err = Pfi * err
Pfo_err = Pfo * err

# transmission from power meter measurement with beam splitter
T_pwmb = Pfo / Pbsr / 2
T_pwmb_err = T_pwmb * np.sqrt(2 * err**2)

# transmission from power meter measurement fiber only
T_pwmf = Pfo / Pfi
T_pwmf_err = T_pwmf * np.sqrt(2 * err**2)

# find average calibration for ADC
cal1_mean, cal1_sem = get_mean(Pbsr / Ch1)
cal2_mean, cal2_sem = get_mean(Pfo / Ch2)

# tranmission from ADC
T_adc = (Ch2 * cal2_mean) / (Ch1 * cal1_mean) / 2
T_adc_err = T_adc * np.sqrt((cal2_sem / cal2_mean)**2 + (cal1_sem / cal1_mean)**2)

# beamsplitter ratio of reflected over transmitted
R_bs = Pbsr / Pbst
R_bs_err = R_bs * np.sqrt(2 * err**2)

# ADC calibration constants
cal1 = Pbsr / Ch1
cal1_err = cal1 * err
cal2 = Pfo / Ch2
cal2_err = cal2 * err

# print the numerical results
file = 'Output/PM_results.txt'
open(file, 'w')

tprint(f'Ch1 average calibration constant    = {cal1_mean:.4g} ± {cal1_sem:.2g} mW V^-1')
tprint(f'Ch2 average calibration constant    = {cal2_mean:.4g} ± {cal2_sem:.2g} mW V^-1')


# parameters for plotting fiber transmission
plt.figure(1)
plt.title('Fiber Transmission over Time', pad=40)
plt.xlabel('time $t$ (h)')
plt.ylabel(r'transmission $T$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

# plot the transmission ratios
plt.errorbar(t, T_pwmb, yerr=T_pwmb_err, fmt='.', capsize=5, c=clr[0], label='$T_{power \; meter}$')
plt.errorbar(t, T_pwmf, yerr=T_pwmf_err, fmt='.', capsize=5, c=clr[1], label='$T_{fiber \; only}$')
plt.errorbar(t, T_adc, yerr=T_adc_err, fmt='.', capsize=5, c=clr[2], label='$T_{ADC}$')

plt.legend(loc=(0.1, 1.05), ncol=3)

# save plot
plt.savefig('Output/PM_transmission.png', dpi=300, bbox_inches='tight')

# parameters for plotting beam splitter ratio
plt.figure(2)
plt.title('Beamsplitter Ratio over Time')
plt.xlabel('time $t$ (h)')
plt.ylabel('beamsplitter ratio $R_{R/T}$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

# plot the beamsplitter ratio
plt.errorbar(t, R_bs, yerr=R_bs_err, fmt='.', capsize=5, c=clr[0])

# save plot
plt.savefig('Output/PM_beamsplitter.png', dpi=300, bbox_inches='tight')
plt.show()

# parameters for plotting total laser output power
plt.figure(3)
plt.title('Total Laser Output Power over Time')
plt.xlabel('time $t$ (h)')
plt.ylabel('output power $P_{out}$ (mW)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

# plot the total laser output power
plt.errorbar(t, Ptot, yerr=Ptot_err, fmt='.', capsize=5, c=clr[0])

# save the plot
plt.savefig('Output/PM_power.png', dpi=300, bbox_inches='tight')

# parameters for plotting
plt.figure(4)
plt.title('Photodiode Degradation over Time', pad=40)
plt.xlabel('time $t$ (h)')
plt.ylabel('$P_{ADC} / P_{power \; meter}$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

# plot the ADC calibration constants
plt.errorbar(t, cal1, yerr=cal1_err, fmt='.', capsize=5, c=clr[0], label='ADC Ch1 (beamsplitter reflection)')
plt.errorbar(t, cal2, yerr=cal2_err, fmt='.', capsize=5, c=clr[2], label='ADC Ch2 (fiber output)')

plt.legend(loc=(-0.1, 1.05), ncol=2)

# save plot
plt.savefig('Output/PM_calibration.png', dpi=300, bbox_inches='tight')

# show plots
plt.show()