# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 22:31:41 2022

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
tf = m.FV(value=1.0,lb=50,ub=final_time) #FV is a fixed value variable
tf.STATUS = 1 #STATUS = 1 means the optimzer can minimize the function

# Parameters of the problem
G =  m.Const(6.674*10**(-11), name='G') #Gravitational Constant
M =  m.Const(7.346*10**(22), name='M')  #Mass of the Moon
R0 =  1738100       #Radius of the lunar surface

Ft =  m.Const(15346, name='Ft')          #thrust force of engine
M0 =  m.Const(4821, name='M0')          #Wet Mass of rocket
M_dot =  m.Const(5.053, name='M_dot')       #Mass flow rate of propellant

Rfmin = m.Const(86904+R0, name ='Rfmin')  #Minimum final orbital radius
Rfmax = m.Const(106217+R0, name ='Rfmax')  #Maximum final orbital radius

# Position variables
y = m.Var(name='y', value = R0)
ydot = m.Var( name='ydot')
ydoubledot = m.Var(name='ydoubledot')

'''
x = m.Var(value=1, name='x')
xdot = m.Var(value = 100, name='xdot')
xdoubledot = m.Var(value = 1.5, name='xdoubledot')'''

mass = m.Var(value=4821, lb = 2245, ub = M0, name='mass') #lb, ub are upper and lower bounds respectively

#theta = m.Var(name='theta', lb = -np.pi, ub = np.pi) #May require to be changed to MV

angle = m.MV(name='angle', value = 0, lb=-1, ub=1) #Idealy, this would be replaced by a trig function
angle.DPRED

angle.STATUS = 1 #Allows computer to change theta
angle.DCOST = 1e-5 #Adds a very small cost to changes in theta
angle.REQONCTRL = 1





# differential equations scaled by tf
m.Equation(y.dt()==tf*ydot) #Expression for y velocity
m.Equation(ydot.dt() == tf*ydoubledot) #Expression for y acceleration

m.Equation(mass.dt() == -M_dot*tf) #Expression for mass
#m.Equation(theta.dt() == thetadot*tf) 

#System dynamics:
#m.Equation(angle == m.cos(theta) )
'''
m.Equation(angle >= 0) 
m.Equation(angle <= 3.14159) '''

m.Equation(ydoubledot == (Ft/mass)*angle-((G*M)/(y**2)))    


#Constraints
m.Equation(y >= R0)


#Creates a list that satisfies final condition at all time except at final time for constraint_1
constraint_1 = np.full(nt, R0+20500)
constraint_1 [-1] = 0

final_radius = m.Param(value = constraint_1 )
#Make sure final orbital radius is greater than Rfmin
m.Equation(y+final_radius >= R0+20000)

#Creates a list that satisfies final condition at all time except at final time for constraint_2
constraint_2 = np.full(nt, 0)
constraint_2 [-1] = 1
final_velocity = m.Param(value = constraint_2 )

m.Equation(ydot*final_velocity <= 100)
#m.Equation(m.cos(theta)*final_velocity <= 0.2)
#m.Maximize(ydot*final_velocity)
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
ax.plot(ts,angle.value)
ax.set_title('theta')
ax.set_xlabel('Time, sec')
ax.grid()

plt.show()


'''
m.Equation(x.dt()==tf*xdot) #Expression for x velocity
m.Equation(xdot.dt() == tf*xdoubledot) #Expression for x acceleration
'''

'''
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
'''