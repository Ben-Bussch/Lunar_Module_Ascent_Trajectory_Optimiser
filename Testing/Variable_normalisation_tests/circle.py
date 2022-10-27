# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:58:08 2022

@author: Bobke
"""

import numpy as np
import matplotlib.pyplot as plt

angle = np.linspace(0, 2*np.pi, 100)
radius = 1738100/86904
a = radius*np.cos(angle)
b = radius*np.sin(angle)

radius = 173810/86904
c = radius*np.cos(angle)
d = radius*np.sin(angle)

fig6 = plt.figure()
ax = fig6.add_subplot()
ax.plot(a, b)
ax.plot(c,d)
ax.set_title('Position')
plt.ylabel('y')
ax.set_xlabel('x')
ax.grid()
ax.set_aspect('equal')
plt.savefig('tae.png', dpi=300)

x = np.zeros(10)

print(x[0])