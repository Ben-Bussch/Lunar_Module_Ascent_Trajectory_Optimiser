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

nt = 301 #number of timesteps
# scale 0-1 time with tf
m.time = np.linspace(0,1,nt)

# options
m.options.NODES = 4 #The number of collocation points between each timestep
m.options.SOLVER = 3 #1 = APOPT, 3 = IPOPT 
m.options.IMODE = 6  #Tells gekko the problem is an optimal control problem
m.options.MAX_ITER = 1000 
m.options.MV_TYPE = 0 #How the interpolation between Manipulated variables (MVs) is done
#m.options.COLDSTART= 2 #Should help find bad constraint
m.options.OTOL = 1e-2 #Default of 1e-6, but should be a helping start
m.options.RTOL = 1e-2
m.options.DIAGLEVEL = 2

# final time
final_time = 470 #470
tf = m.FV(value=1.0,lb=0.1,ub=final_time) #FV is a fixed value variable
tf.STATUS = 1 #STATUS = 1 means the optimzer can minimize the function

# Parameters of the problem
G =  m.Const(6.674*10**(-11), name='G') #Gravitational Constant
M =  m.Const(7.346*10**(22), name='M')  #Mass of the Moon
R0 =  m.Const(1738100, name='R0')       #Radius of the lunar surface

Ft =  m.Const(15346, name='Ft')          #thrust force of engine
M0 =  m.Const(4821, name='M0')          #Wet Mass of rocket
M_dot =  m.Const(5.053, name='M_dot')       #Mass flow rate of propellant

Rfmin = m.Const(86904+R0, name ='Rfmin')  #Minimum final orbital radius
Rfmax = m.Const(106217+R0, name ='Rfmax')  #Maximum final orbital radius


# Position variables
y = m.Var(name='y')
ydot = m.Var( name='ydot')
ydoubledot = m.Var(name='ydoubledot')

x = m.Var(value=1, name='x')
xdot = m.Var(value = 100, name='xdot')
xdoubledot = m.Var(value = 1.5, name='xdoubledot')

mass = m.Var(value=4821, lb = 2245, ub = M0, name='mass') #lb, ub are upper and lower bounds respectively
theta = m.MV(name='theta', lb = -3.1415926, ub = 3.1415926) #May require to be changed to MV
#theta.STATUS = 1



# differential equations scaled by tf
m.Equation(y.dt()==tf*ydot) #Expression for x velocity
m.Equation(ydot.dt() == tf*ydoubledot) #Expression for x acceleration
m.Equation(x.dt()==tf*xdot) #Expression for y velocity
m.Equation(xdot.dt() == tf*xdoubledot) #Expression for y acceleration

m.Equation(mass.dt() == -M_dot*tf) #Expression for mass


#System dynamics:
m.Equation(xdoubledot == ( (Ft/(mass*(x**2 + y**2)**(1/2)))*(x*m.cos(theta)-y*m.sin(theta)) )\
           - x*(G*M/((x**2 + y**2)**(3/2)))\
           )
                        
m.Equation(ydoubledot == ( (Ft/(mass*(x**2 + y**2)**(1/2)))*(y*m.cos(theta)+x*m.sin(theta)) )\
           - y*(G*M/((x**2 + y**2)**(3/2)))\
           )



#Initial Boundary Conditions:
  
m.fix(y, pos=0,val=R0)      #Initial y position
m.fix(x, pos=0,val=0)       #Initial x position
m.fix(ydot, pos=0,val=0)    #Initial y velocity
m.fix(xdot, pos=0,val=0)    #Initial x velocity

m.fix(theta, pos=0, val=0)  #Initial angle
m.fix(mass, pos=0,val=4821) #Initial mass  


# Final Boundary Conditions

#Making sure the vector components of position and the dyanmics are not bound
m.free_final(theta)
m.free_final(y)
m.free_final(ydot)
m.free_final(ydoubledot)
m.free_final(x)
m.free_final(xdot)
m.free_final(xdoubledot)

#Creates a list that satisfies final condition at all time except at final time for constraint_1
constraint_1 = np.full(nt, Rfmin+1)
constraint_1 [-1] = 0
final_radius = m.Param(value = constraint_1 )
#Creates a list that satisfies final condition at all time except at final time for constraint_2
constraint_2 = np.full(nt, 0)
constraint_2 [-1] = 1
final_dot = m.Param(value = constraint_2 )

#Make sure final orbital radius is greater than Rfmin
m.Equation(y**2+x**2+final_radius**2 >= Rfmin**2)

#Makes sure the final velocity is greater than or equal to the orbital velocity
m.Equation(((xdot**2 + ydot**2)**(1/2))*final_dot >= G*M*final_dot/((x**2+y**2)**(1/2)))
m.Minimize(((xdot**2 + ydot**2)**(1/2))*final_dot)

#Minimizes dot product of velocity and position
m.Equation(y*ydot*final_dot+x*xdot*final_dot >= 0)
m.Minimize(y*ydot*final_dot+x*xdot*final_dot)

# minimize final time
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
plt.plot(ts,y.value)
ax.set_title('y, m')
ax.set_xlabel('Time, sec')
ax.grid()

fig2 = plt.figure()
ax = fig2.add_subplot()
plt.plot(ts,ydot.value)
ax.set_title('Speed')
ax.set_xlabel('Time, sec')
ax.grid()

fig3 = plt.figure()
ax = fig3.add_subplot()
plt.plot(ts,ydoubledot.value)
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
ax.plot(ts,theta.value)
ax.set_title('pheta')
ax.set_xlabel('Time, sec')
ax.grid()

plt.show()