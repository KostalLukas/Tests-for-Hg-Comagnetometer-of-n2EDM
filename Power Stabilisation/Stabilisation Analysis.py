# -*- coding: utf-8 -*-
"""
Power Stabilisation Analysis v2.0

Lukas Kostal, 28.7.2023, PSI
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate as si
import sys


# ignore all warnings from numpy
np.seterr(all="ignore")


# function to read arguments from console
def garg(*args):
    args = list(args)
    arg_sys = sys.argv
    for i in range(1, len(arg_sys)):
        if type(args[i-1]) == bool:
            args[i-1] = arg_sys[i] == 'True'
        else:
            args[i-1] = type(args[i-1])(arg_sys[i])
    return args


# function to print to console and file simultaneously
def tprint(text=''):
    print(text)
    global file
    with open(file, 'a') as output:
        print(text, file=output)
        

# data to be analysed
data = 'Modulation_VIP'

# setpoint power in mW
Pset = 2000
# TA current at Vmod=0 in mA
Iact = 3086
# voltage limits on PID controller in V
Vmax = 1
Vmin = -1

# relative error in power meter measurements
err = 0.05

# sensitivity of TA modultion input in mA per V
sens = 200

# specify or input filename of data to be analyzed
data, Pset, Iact, Vmac, Vmin = garg(data, Pset, Iact, Vmax, Vmin)

# specify colors for plotting
colr = ['royalblue', 'green', 'orange', 'red']

# load the data
vals = np.loadtxt(f'Data/{data}.csv', unpack=True, delimiter=',', skiprows=1)

# slice the values to get I and P arrays
I = vals[-2, :]
P = vals[-1, :]

# convert power from mW to uW and calcualte absolute random error
P *= 1e3
Perr = err * P

# check if data also contains V
if len(vals) == 3:
    V = vals[0, :]
    
    # get indices for which P and delte the corresponding datapoints
    i_nz = np.nonzero(P)
    V_nz = V[i_nz]
    I_nz = I[i_nz]
    P_nz = P[i_nz]
    
    # get indces to sort datapoints in increasing P
    i_ord = np.argsort(P_nz)
    
    # interpolate with cubic splines
    get_IfromP = si.CubicSpline(P_nz[i_ord], I_nz[i_ord])
    get_PfromI = si.CubicSpline(I, P)
    
    get_VfromP = si.CubicSpline(P_nz[i_ord], V_nz[i_ord])
    get_VfromI = si.CubicSpline(I, V)

    Iact_max = Iact + sens * Vmax
    Iact_min = Iact + sens * Vmin 
    Pact     = get_PfromI(Iact)
    Pact_max = get_PfromI(Iact_max)
    Pact_min = get_PfromI(Iact_min)

    Iset     = get_IfromP(Pset)
    Iset_max = Iset + sens * Vmax
    Iset_min = Iset + sens * Vmin
    Pset_max = get_PfromI(Iset_max)
    Pset_min = get_PfromI(Iset_min)
    
    # check if the Iact range is within dataset range
    if Iact_max > np.amax(I) or Iact_min < np.amin(I) :
        print('Warning: Iact TA current modulation range is outside the range of the dataset')
        
    # check if the Pset value is within dataset range
    if Pset > np.amax(P) or Pset < np.amin(P):
        print('Warning: Pset desired setpoint is outside the range of the dataset')    
    
    # check if the Iset required for the Pset is within the dataset range
    elif Iset_max > np.amax(I) or Iset_min < np.amin(I):
        print('Warning: Current modulation required for Pset is outside the range of the dataset')

    # print the numerical results
    file = 'Output/VIP_results.txt'
    open(file, 'w')
    
    tprint('Modulation limits:')
    tprint(f'Vmax       = {Vmax:.4g} V')
    tprint(f'Vmin       = {Vmin:.4g} V')
    tprint()
    tprint('Actual parameters:')
    tprint(f'Iact       = {Iact:.4g} mA')
    tprint(f'Iact_max   = {Iact_max:.4g} mA')
    tprint(f'Iact_min   = {Iact_min:.4g} mA')
    tprint()
    tprint(f'Pact       = {Pact:.4g} uW')
    tprint(f'Pact_max   = {Iact_max:.4g} uW')
    tprint(f'Pact_min   = {Iact_min:.4g} uW')
    tprint()
    tprint('Setpoint Patameters:')
    tprint(f'Iset       = {Iset:.4g} mA')
    tprint(f'Iset_max   = {Iset_max:.4g} mA')
    tprint(f'Iset_min   = {Iset_min:.4g} mA')
    tprint()
    tprint(f'Pset       = {Pset:.4g} uW')
    tprint(f'Pset_max   = {Iset_max:.4g} uW')
    tprint(f'Pset_min   = {Iset_min:.4g} uW')

    # parameters for plotting TA current against modulation voltage
    fig1 = plt.figure(1)
    fig1.set_tight_layout(True)

    plt.title('TA Current against Modulation Voltage')
    plt.xlabel('modulation voltage $V{mod}$ (V)')
    plt.ylabel('TA current $I_{act}$ (mA)')
    plt.rc('grid', linestyle=':', c='black', alpha=0.8)
    plt.grid()
    
    plt.plot(V, I, 'x', c=colr[1])

    # save plot
    plt.savefig('Output/Modulation_VI.png', dpi=300, bbox_inches='tight')
    
    # parameters for plotting
    fig2 = plt.figure(2)
    fig2.set_tight_layout(True)
    
    plt.title('Output Power against Modulation Voltage')
    plt.xlabel('modulation voltage $V_{mod}$ (V)')
    plt.ylabel('output power $P$ ($\mu W$)')
    plt.rc('grid', linestyle=':', c='black', alpha=0.8)
    plt.grid()

    plt.errorbar(V, P, yerr=Perr, fmt='x', capsize=3, c=colr[0])

    # save plot
    plt.savefig('Output/Modulation_VP.png', dpi=300, bbox_inches='tight')
    
    # parameters for plotting output power against TA current
    fig3 = plt.figure(3)
    fig3.set_tight_layout(True)
    
    plt.title('Output Power against TA Current', pad=40)
    plt.xlabel('TA current $I_{act}$ (mA)')
    plt.ylabel('output power $P$ ($\mu W$)')
    plt.rc('grid', linestyle=':', c='black', alpha=0.8)
    plt.grid()

    plt.errorbar(I, P, yerr=Perr, fmt='x', capsize=3, c=colr[0], \
                 label='measurement')
    plt.axvline(Iact_max, lw=1.2, c=colr[0])
    plt.axvline(Iact_min, lw=1.2, c=colr[0])
    plt.axhline(Pact_max, lw=1.2, c=colr[0])
    plt.axhline(Pact_min, lw=1.2, c=colr[0])
    plt.fill_between([Iact_min, Iact_max], [Pact_min, Pact_min], \
                     [Pact_max, Pact_max], color=colr[0], alpha=0.2, \
                     label='actual mod. region')
    
    plt.axvline(Iset_max, lw=1.2, c=colr[3])
    plt.axvline(Iset_min, lw=1.2, c=colr[3])
    plt.axhline(Pset_max, lw=1.2, c=colr[3])
    plt.axhline(Pset_min, lw=1.2, c=colr[3])
    plt.fill_between([Iset_min, Iset_max], [Pset_min, Pset_min], \
                     [Pset_max, Pset_max], color=colr[3], alpha=0.2, \
                     label='desired mod. region')

    # save plot
    plt.legend(loc=(-0.1, 1.05), ncol=3)
    plt.savefig('Output/Modulation_IP.png', dpi=300, bbox_inches='tight')

    # show the plots
    plt.show()

else:
    
    # parameters for plotting output power against TA current
    fig1 = plt.figure(1)
    fig1.set_tight_layout(True)
    
    plt.title('Output Power against TA Current')
    plt.xlabel('TA current $Iact$ (mA)')
    plt.ylabel('output power $P$ ($\mu W$)')
    plt.rc('grid', linestyle=':', c='black', alpha=0.8)
    plt.grid()

    plt.errorbar(I, P, yerr=Perr, fmt='x', capsize=3, c=colr[0])

    plt.savefig('Output/Modulation_IP.png', dpi=300, bbox_inches='tight')

    # show the plot
    plt.show()

    