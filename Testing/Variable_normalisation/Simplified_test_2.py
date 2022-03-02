# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 19:04:27 2022

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
m.options.NODES = 2 #The number of collocation points between each timestep
m.options.SOLVER = 3 #1 = APOPT, 3 = IPOPT 
m.options.IMODE = 6  #Tells gekko the problem is an optimal control problem
m.options.MAX_ITER = 20000 
m.options.MV_TYPE = 0 #How the interpolation between Manipulated variables (MVs) is done
#m.options.COLDSTART= 2 #Should help find bad constraint
m.options.OTOL = 1e-3 #allows for margins of error in solution, default of 1e-6
m.options.RTOL = 1e-3 #same as above
m.options.DIAGLEVEL = 0

# final time
final_time = 470 #470
tf = m.FV(value=0,lb=0,ub=1) #FV is a fixed value variable
tf.STATUS = 1 #STATUS = 1 means the optimzer can minimize the function


# Position variables
y = m.Var(name='y', value = 0)
ydot = m.Var( name='ydot')
ydoubledot = m.Var(name='ydoubledot')


x = m.Var(name='x', value = 0)
xdot = m.Var( name='xdot')
xdoubledot = m.Var(name='xdoubledot')


angle = m.MV(name='angle', value = 0.5, lb = 0, ub = np.pi/3)
#angle.DPRED
angle.STATUS = 1 #Allows computer to change theta
#angle.DCOST = 1e-5 #Adds a very small cost to changes in theta
angle.REQONCTRL = 3 #tells solver whether to change MVs or run as simulator

# differential equations scaled by tf
m.Equation(y.dt()==tf*ydot*final_time) #Expression for y velocity
m.Equation(ydot.dt() == tf*ydoubledot*final_time) #Expression for y acceleration

m.Equation(x.dt()==tf*xdot*final_time) #Expression for x velocity
m.Equation(xdot.dt() == tf*xdoubledot*final_time) #Expression for x acceleration

#system dynamics
m.Equation(ydoubledot == (3.9*m.cos(3*angle)-1)/10000)
m.Equation(xdoubledot == (3.9*m.sin(3*angle)/10000))

#Creates a list that satisfies final condition at all time except at final time for constraint_1
constraint_1 = np.full(nt, 10)
constraint_1 [-1] = 0

final_radius = m.Param(value = constraint_1 )
#Make sure final orbital radius is greater than Rfmin
m.Equation(y+final_radius >= 1)

#Creates a list that satisfies final condition at all time except at final time for constraint_2
constraint_2 = np.full(nt, 0)
constraint_2 [-1] = 1
final_velocity = m.Param(value = constraint_2 )


m.Equation(ydot*final_velocity <= 0) #Enbsures minimal vertical component of velocity
#m.Equation(xdot*final_velocity > 1)

m.Maximize(xdot)
m.Minimize(tf)

# Optimize launch
try:
    m.solve(disp=True)    # solve
except:
    print('Not successful')
    from gekko.apm import get_file
    print(m._server)
    print(m._model_name)
    f = get_file(m._server,m._model_name,'infeasibilities.txt')
    f = f.decode().replace('\r','')
    with open('infeasibilities.txt', 'w') as fl:
        fl.write(str(f))



print('Optimal Solution (final time): ' + str(tf.value[0]))

# scaled time
ts = m.time * tf.value[0]

# plot results
fig1 = plt.figure()
ax = fig1.add_subplot()
plt.plot(ts*470,y.value)
ax.set_title('y, m')
ax.set_xlabel('Time, sec')
ax.grid()

fig2 = plt.figure()
ax = fig2.add_subplot()
plt.plot(ts*470,ydot.value)
ax.set_title('Speed')
ax.set_xlabel('Time, sec')
ax.grid()

fig3 = plt.figure()
ax = fig3.add_subplot()
plt.plot(ts*470,ydoubledot.value)
ax.set_title('Acceleration')
ax.set_xlabel('Time, sec')
ax.grid()


'''fig4 = plt.figure()
ax = fig4.add_subplot()
plt.plot(ts*470,mass.value)
plt.ylabel('Mass')
ax.set_xlabel('Time, sec')
ax.grid()'''

fig5 = plt.figure()
ax = fig5.add_subplot()
ax.plot(ts,angle.value)
ax.set_title('theta')
ax.set_xlabel('Time, sec')
ax.grid()

fig6 = plt.figure()
ax = fig6.add_subplot()
plt.plot(x.value,y.value)
plt.ylabel('y')
ax.set_xlabel('x')
ax.grid()

plt.show()