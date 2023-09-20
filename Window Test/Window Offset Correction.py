# -*- coding: utf-8 -*-
"""
Correction for discontinuity in Window_083114_W7 v1.0

Lukas Kostal, 13.9.2023, PSI
"""


import numpy as np
from matplotlib import pyplot as plt


# correction by which the first section is offset
corr = 6.9e-3

# load the data
V_arr = np.genfromtxt('Data/Window_083114_W7.txt', skip_footer=1, unpack=True, delimiter=',')

# plot the first part of the data when laser had poor stability
plt.plot(V_arr[1, :])
plt.ylim(0.541, 0.545)
plt.xlim(80000, 84000)
plt.show()

# plot the second part of the data when laser regained stability
plt.plot(V_arr[1, :])
plt.ylim(0.535, 0.537)
plt.xlim(86000, 88000)
plt.show()

# offset the first section on Ch2
V_arr[1, :86000] -= 6.9e-3

# write the corrected data into a file
file = 'Data/Window_083114_W7_Cr.txt'
out = open(file, 'w')

for i in range(0, len(V_arr[0, :])):
    with open(file, 'a') as output:
        print(f'{V_arr[0, i]:#.6f}, {V_arr[1, i]:#.6f}, {V_arr[2, i]:#.6f}, {V_arr[3, i]:#.6f}', file=output)
    