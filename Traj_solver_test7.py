# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:27:54 2022

@author: Bobke
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:21:23 2022

@author: Bobke
"""
import matplotlib.pyplot as plt
import numpy as np

import shutil
import sys
import os.path

assert(shutil.which("ipopt") or os.path.isfile("ipopt"))
from pyomo.environ import *
from pyomo.dae import *

# lunar module
m_ascent_dry = 2445.0          # kg mass of ascent stage without fuel
m_ascent_fuel = 2376.0         # kg mass of ascent stage fuel
m_descent_dry = 2034.0         # kg mass of descent stage without fuel
m_descent_fuel = 8248.0        # kg mass of descent stage fuel

R0 = 1738100
Rfmin = 86904.6+R0

m_fuel = m_ascent_fuel
m_dry = m_ascent_dry
m_total = m_dry + m_fuel

# descent engine characteristics
v_exhaust = 3040.0             # m/s
u_max = 5.053      # 45050 newtons / exhaust velocity

f_thrust = 15246
mass_flow = u_max

# landing mission specifications
h_initial = 100000.0           # meters
v_initial = 1520               # orbital velocity m/s
g = 1.62                       # m/s**2
m = ConcreteModel()
m.t = ContinuousSet(bounds=(0, 1))
m.x = Var(m.t, domain=NonNegativeReals)
m.m = Var(m.t)
m.u = Var(m.t, bounds=(u_max, u_max))
m.T = Var(bounds=(50,470))

m.xdot = DerivativeVar(m.x, wrt=m.t)
m.xdoubledot = DerivativeVar(m.xdot, wrt=m.t)
m.mdot = DerivativeVar(m.m, wrt=m.t)

m.fuel = Integral(m.t, wrt=m.t, rule = lambda m, t: m.u[t]*m.T)
#m.fuel = mass_flow*m.T
m.obj = Objective(expr=m.fuel, sense=minimize)

m.ode1 = Constraint(m.t, rule = lambda m, t: m.m[t]*m.xdoubledot[t]/m.T**2 == -m.m[t]*g + m.u[t]*v_exhaust)
m.ode2 = Constraint(m.t, rule = lambda m, t: m.mdot[t]/m.T == -m.u[t])

#Final orbit height
m.rad_final = Constraint(m.t, rule = lambda m, t: Constraint.Skip if t != m.t.last() else \
                         m.x[t] >= Rfmin)


    
m.x[0].fix(R0)
m.xdot[0].fix(0)
m.m[0].fix(m_total)

#m.h[1].fix(Rfmin)    # land on surface
#m.v[1].fix(-v_initial)    # soft landing

def solve(m):
    TransformationFactory('dae.finite_difference').apply_to(m, nfe=50, scheme='FORWARD')
    
    solver = SolverFactory('ipopt')
    solver.options["halt_on_ampl_error"] = "yes" 
    results = solver.solve(m, tee=True)
    
    
    m_nonfuel = m_ascent_dry 
    
    tsim = [t*m.T() for t in m.t]
    hsim = [m.x[t]() for t in m.t]
    asim = [m.xdoubledot[t]() for t in m.t]
    usim = [m.u[t]() for t in m.t]
    fsim = [m.m[t]()-m_nonfuel for t in m.t]

    plt.figure(figsize=(8,6))
    plt.subplot(3,1,1)
    plt.plot(tsim, hsim)
    plt.title('altitude')
    plt.ylabel('meters')
    plt.legend(['mission length = ' + str(round(m.T(),1)) + ' seconds'])
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(tsim,  asim)
    plt.title('acceleration')
    plt.ylabel('ms-2')
    #plt.legend(['fuel burned = ' + str(round(m.fuel(),1)) + ' kg'])
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(tsim, fsim)
    plt.title('fuel remaining')
    plt.xlabel('time / seconds')
    plt.ylabel('kg')
    plt.legend(['fuel remaining = ' + str(round(fsim[-1],2)) + ' kg'])
    plt.grid(True)

    plt.tight_layout()

solve(m)
