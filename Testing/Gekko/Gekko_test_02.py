# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 14:39:07 2022

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO

# create GEKKO model
m = GEKKO()

nt = 101 #number of timesteps
# scale 0-1 time with tf
m.time = np.linspace(0,1,nt)

# options
m.options.NODES = 6 #The number of collocation points between each timestep
m.options.SOLVER = 3 #Solver = 3 means IPOPT is used as the solver
m.options.IMODE = 6  #Tells gekko the problem has dynamics and is an optimal control problem
m.options.MAX_ITER = 500
m.options.MV_TYPE = 0 #How the interpolation between Manipulated variables (MVs) is done
m.options.DIAGLEVEL = 2

# final time
final_time = 10000 #470
tf = m.FV(value=1.0,lb=0.1,ub=final_time) #FV is a fixed value variable
tf.STATUS = 1 #STATUS = 1 means the optimzer can minimize the function

# Parameters of the problem
G =  m.Const(6.674*10**(-11)) #Gravitational Constant
M =  m.Const(7.346*10**(22))  #Mass of the Moon
R0 =  1738100       #Radius of the lunar surface

Ft =  m.Const(15346)          #thrust force of engine
M0 =  m.Const(4821)          #Wet Mass of rocket
M_dot =  m.Const(5.053)       #Mass flow rate of propellant

Rfmin = 86904+R0  #Minimum final orbital radius
Rfmax = 106217+R0   #Maximum final orbital radius


# Position variables
x = m.Var(value=R0)
xdot = m.Var()
xdoubledot = m.Var()

y = m.Var(value=0)
ydot = m.Var()
ydoubledot = m.Var()


mass = m.Var(value=4821) #lb, ub are upper and lower bounds respectively
pheta = m.MV(lb=-np.pi,ub=np.pi)

pheta.STATUS = 1

#Gravity
#Fg = m.Intermediate(G*M/x**2)


# differential equations scaled by tf
m.Equation(x.dt()==tf*xdot) #Expression for x velocity
m.Equation(xdot.dt() == tf*xdoubledot) #Expression for x acceleration

m.Equation(y.dt()==tf*ydot) #Expression for y velocity
m.Equation(ydot.dt() == tf*ydoubledot) #Expression for y acceleration


m.Equation(ydoubledot == (Ft/mass)*m.cos(pheta) - 1.6)
m.Equation(xdoubledot == (Ft/mass)*m.sin(pheta))
m.Equation(mass.dt() == -tf*M_dot) #Expression for mass


#Initial Boundary Conditions:
m.fix(mass, pos=0,val=4821) #Initial mass 
   
m.fix(x, pos=0,val=0) #Initial x position
m.fix(y, pos=0,val=R0) #Initial y position   

m.fix(xdot, pos=0,val=0) #Initial x velocity
m.fix(ydot, pos=0,val=0) #Initial y velocity


# Final Boundary Conditions

#Creates a list that satisfies final condition at all time except at final time
constraint_1 = np.full(nt, Rfmin+1)
constraint_1 [-1] = 0
final_y = m.Param(value = constraint_1 )

#At final time, rocket position should be greater than minimum radius
m.Equation(y+final_y >= Rfmin)

#m.Equation(y*final_x >= Rfmin**2)


'''
#Creates a list that satisfies final condition at all time except at final time
constraint_2 = np.full(nt, 0)
constraint_2 [-1] = 1.0
final_velocity_direction = m.Param(value = constraint_2 )

#At final time, position and velocity dto product should be zero
m.Equation((x*xdot+y*ydot)*final_velocity_direction <= 10)
m.Equation((x*xdot+y*ydot)*final_velocity_direction >= 10)
'''

# minimize final time
m.Obj(tf)

# Optimize launch
m.solve(disp=True)



print('Optimal Solution (final time): ' + str(tf.value[0]))

# scaled time
ts = m.time * tf.value[0]

# plot results
fig1 = plt.figure()
ax = fig1.add_subplot()
plt.plot(ts,x.value)
ax.set_title('x, m')
ax.set_xlabel('Time, sec')
ax.grid()

fig2 = plt.figure()
ax = fig2.add_subplot()
plt.plot(ts,xdot.value)
ax.set_title('Speed')
ax.set_xlabel('Time, sec')
ax.grid()

fig3 = plt.figure()
ax = fig3.add_subplot()
plt.plot(ts,xdoubledot.value)
ax.set_title('Acceleration')
ax.set_xlabel('Time, sec')
ax.grid()

fig4 = plt.figure()
ax = fig4.add_subplot()
plt.plot(ts,mass.value)
plt.ylabel('Mass')
ax.set_xlabel('Time, sec')
ax.grid()

fig5 = plt.figure()
ax = fig5.add_subplot()
ax.plot(ts,pheta.value)
ax.set_title('pheta')
ax.set_xlabel('Time, sec')
ax.grid()


'''
fig2 = plt.figure()
ax = fig2.add_subplot()
ax.plot(tdirect, ydirect)
ax.set_title('y, m')
ax.set_xlabel('Time, sec')
ax.grid()



fig4 = plt.figure()
ax = fig4.add_subplot()
ax.plot(tdirect, ydotdirect)
ax.set_title(r'$V_y$, m/s')
ax.set_xlabel('Time, sec')
ax.grid()'''


plt.show()