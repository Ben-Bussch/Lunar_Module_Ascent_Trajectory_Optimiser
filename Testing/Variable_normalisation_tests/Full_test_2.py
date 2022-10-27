# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:13:46 2022

@author: Bobke
"""

import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
from gekko import *

# create GEKKO model
m = GEKKO()

nt = 200 #number of timesteps
# scale 0-1 time with tf
m.time = np.linspace(0,1,nt)

# options
m.options.NODES = 4 #The number of collocation points between each timestep
m.options.SOLVER = 3 #1 = APOPT, 3 = IPOPT 
m.options.IMODE = 6  #Tells gekko the problem is an optimal control problem
m.options.MAX_ITER = 20000 
m.options.MV_TYPE = 0 #How the interpolation between Manipulated variables (MVs) is done

m.options.OTOL = 1e-3 #allows for margins of error in solution, default of 1e-6
m.options.RTOL = 1e-3 #same as above
m.options.DIAGLEVEL = 0

#m.options.COLDSTART= 2 #Should help find bad constraint

# Parameters of the problem
G =  m.Const(6.674*10**(-11), name='G') #Gravitational Constant
M =  m.Const(7.346*10**(22), name='M')  #Mass of the Moon
R0 =  1738100       #Radius of the lunar surface

Ft =  m.Const(15346, name='Ft')          #thrust force of engine
M0 =  m.Const(4821, name='M0')           #Wet Mass of rocket

#Scalars
distance_scalar = 86904
mass_scalar = 2576
GravT_scalar = 1e6

#
orbital_v = ((6.674*10**(-11)*7.346*10**(22))/(1738100+distance_scalar))**(1/2)
x_offset = 0
y_offset =  1738100


# final time
final_time = 470 #470
tf = m.FV(value=0,lb=0,ub=1) #FV is a fixed value variable
tf.STATUS = 1 #STATUS = 1 means the optimzer can minimize the function

#Mass
mflow = 5.053/2376
mass = m.Var(value=0, lb = 0, ub = 1, name='mass') #lb, ub are upper and lower bounds respectively

# State variables
y = m.Var(name='y', value = 0)
ydot = m.Var( name='ydot')
ydoubledot = m.Var(name='ydoubledot')

x = m.Var(name='x', value = 0)
xdot = m.Var( name='xdot')
xdoubledot = m.Var(name='xdoubledot')

angle = m.MV(name='angle', value = 0)
#angle.DPRED
angle.STATUS = 1 #Allows computer to change theta
angle.DCOST = 1e-5 #Adds a very small cost to changes in theta
angle.REQONCTRL = 3 #tells solver whether to change MVs or run as simulator

#intermidiate variables
Tx = m.Var(name='Tx', value = 0)
Ty = m.Var(name='Ty', value = 0)

T = m.Var(name='T', value = 0)
grav = m.Var(name='grav', value = 0)


# differential equations scaled by tf
m.Equation(x.dt()==tf*xdot*final_time) #Expression for x velocity
m.Equation(xdot.dt() == tf*xdoubledot*final_time*distance_scalar) #Expression for x acceleration

m.Equation(y.dt()==tf*ydot*final_time) #Expression for y velocity
m.Equation(ydot.dt() == tf*ydoubledot*final_time*distance_scalar) #Expression for y acceleration


m.Equation(mass.dt() == mflow*final_time*tf) #Expression for mass


#System dynamics:
    
#Grav scaled by 1e6
m.Equation(grav == -((G*M)/((((x*distance_scalar+x_offset)**2) +((y*distance_scalar+y_offset)**2))**(3/2)))*GravT_scalar)

#T scaled by 1e6
m.Equation(T == (Ft/((M0-mass_scalar*mass)*((((x*distance_scalar+x_offset)**2)+((y*distance_scalar+y_offset)**2))**(1/2))))*GravT_scalar )

m.Equation(Tx == (T/GravT_scalar)*((x*distance_scalar+x_offset)*m.cos(3*angle)-(y*distance_scalar+y_offset)*m.sin(3*angle)) ) #Normal scale
m.Equation(Ty == (T/GravT_scalar)*((y*distance_scalar+y_offset)*m.cos(3*angle)+(x*distance_scalar+x_offset)*m.sin(3*angle)) ) #Normal scale


m.Equation(xdoubledot == Tx-((x*distance_scalar+x_offset)*(grav/GravT_scalar)))
m.Equation(ydoubledot == Ty-((y*distance_scalar+y_offset)*(grav/GravT_scalar)))



#Boundary Conditions:
    
#Creates a list that satisfies final condition at all time except at final time for constraint_1
constraint_1 = np.full(nt, 10)
constraint_1 [-1] = 0

final_1 = m.Param(value = constraint_1 )
#Make sure final orbital radius is greater than Rfmin
#m.Equation(y+final_1 >= 0.1)


#Solving:
m.Minimize(tf)
m.solve(disp=True)    # solve
