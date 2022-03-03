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

# Parameters of the problem
G =  m.Const(6.674*10**(-11), name='G') #Gravitational Constant
M =  m.Const(7.346*10**(22), name='M')  #Mass of the Moon
R0 =  m.Const(1738100, name='R0')       #Radius of the lunar surface

Ft =  m.Const(15346, name='Ft')          #thrust force of engine
M0 =  m.Const(4821, name='M0')            #Wet Mass of rocket
M_dot =  m.Const(5.053, name='M_dot')       #Mass flow rate of propellant

Rfmin = m.Const(86904, name ='Rfmin') #final height of orbit



orbital_v = ((6.674*10**(-11)*7.346*10**(22))/(1738100+86904))**(1/2)
#print(orbital_v)
mflow = 5.053/2376

# final time
final_time = 470 #470
tf = m.FV(value=0,lb=0,ub=1) #FV is a fixed value variable
tf.STATUS = 1 #STATUS = 1 means the optimzer can minimize the function

#Mass
mass = m.Var(value=0, lb = 0, ub = 1, name='mass') #lb, ub are upper and lower bounds respectively

# Position variables
y = m.Var(name='y', value = 0)
ydot = m.Var( name='ydot')
ydoubledot = m.Var(name='ydoubledot')

x = m.Var(name='x', value = 0)
xdot = m.Var( name='xdot')
xdoubledot = m.Var(name='xdoubledot')

#Distance scale:
Scalar = m.Const(86904, name = 'distance Scale')
mass_scalar = m.Const(2576, name = 'mass Scale')

angle = m.MV(name='angle', value = 0, lb = 0, ub = np.pi/3)
#angle.DPRED
angle.STATUS = 1 #Allows computer to change theta
angle.DCOST = 1e-5 #Adds a very small cost to changes in theta
angle.REQONCTRL = 3 #tells solver whether to change MVs or run as simulator

# differential equations scaled
m.Equation(y.dt()==tf*ydot*final_time) #Expression for y velocity
m.Equation(ydot.dt() == tf*ydoubledot*final_time) #Expression for y acceleration


m.Equation(x.dt()==tf*xdot*final_time) #Expression for x velocity
m.Equation(xdot.dt() == tf*xdoubledot*final_time) #Expression for x acceleration

m.Equation(mass.dt() == mflow*final_time*tf) #Expression for mass

#system dynamics

m.Equation(ydoubledot == ( ( (Ft/((M0-mass_scalar*mass)*((x*Scalar)**2 + (y*Scalar+R0)**2)**(1/2)))*\
                          ((y*Scalar+R0)*m.cos(3*angle)+(x*Scalar)*m.sin(3*angle)) )\
           - (y*Scalar+R0)*(G*M/(((x*Scalar)**2 + (y*Scalar+R0)**2)**(3/2))) )/ Scalar\
           )


m.Equation(xdoubledot == ( ( (Ft/((M0-mass_scalar*mass)*((x*Scalar)**2 + (y*Scalar+R0)**2)**(1/2)))*\
                          ((x*Scalar)*m.cos(3*angle)-(y*Scalar+R0)*m.sin(3*angle)) )\
           - (x*Scalar)*(G*M/(((x*Scalar)**2 + (y*Scalar+R0)**2)**(3/2))))/ Scalar\
           )

#Initial Boundary Conditions:
  
m.fix(y, pos=0,val=0)       #Initial y position
m.fix(x, pos=0,val=0)       #Initial x position
m.fix(ydot, pos=0,val=0)    #Initial y velocity
m.fix(xdot, pos=0,val=0)    #Initial x velocity

m.fix(angle, pos=0, val=0)  #Initial angle
m.fix(mass, pos=0,val=0) #Initial mass     
    
#Creates a list that satisfies final condition at all time except at final time for constraint_1
constraint_1 = np.full(nt, Rfmin+R0+1)
constraint_1 [-1] = 0

final_radius = m.Param(value = constraint_1 )
#Make sure final orbital radius is greater than Rfmin
#m.Equation(y+final_radius >= 1)

#m.Equation((y*Scalar+R0)**2+(x*Scalar+R0)**2+final_radius**2 >= (Rfmin+R0)**2) #ensures final radius is greater or equal to rfmin
m.Equation(((y+R0/Scalar)**2+(x)**2)**(1/2)+final_radius >= ((R0+Scalar)/Scalar))


#Creates a list that satisfies final condition at all time except at final time for constraint_2
constraint_2 = np.full(nt, 0)
constraint_2 [-1] = 1
final_velocity = m.Param(value = constraint_2 )


m.Equation((xdot**2 + ydot**2) >= (orbital_v/Scalar)**2*final_velocity)#ensures final velocity is orbital


#m.Equation(ydot*final_velocity*Scalar <= 0) #Ensures minimal vertical component of velocity
#Minimizes dot product of velocity and position
m.Equation(((y*Scalar+R0)*(ydot*Scalar)+(x*Scalar)*(xdot*Scalar))*final_velocity == 0)

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

'''
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
ax.set_title('Y Speed')
ax.set_xlabel('Time, sec')
ax.grid()

fig3 = plt.figure()
ax = fig3.add_subplot()
plt.plot(ts*470,ydoubledot.value)
ax.set_title('Y Acceleration')
ax.set_xlabel('Time, sec')
ax.grid()


fig4 = plt.figure()
ax = fig4.add_subplot()
plt.plot(ts*470,mass.value)
plt.ylabel('Mass')
ax.set_xlabel('Time, sec')
ax.grid()

fig5 = plt.figure()
ax = fig5.add_subplot()
ax.plot(ts,angle.value)
ax.set_title('theta')
ax.set_xlabel('Time, sec')
ax.grid()

fig6 = plt.figure()
ax = fig6.add_subplot()
plt.plot(x.value,y.value)
ax.set_title('Position')
plt.ylabel('y')
ax.set_xlabel('x')
ax.grid()

fig7 = plt.figure()
ax = fig7.add_subplot()
plt.plot(ts*470,xdot.value)
ax.set_title('X Speed')
plt.ylabel('x')
ax.set_xlabel('time')
ax.grid()

fig8 = plt.figure()
ax = fig8.add_subplot()
plt.plot(ts*470,xdoubledot.value)
ax.set_title('X Acceleration')
plt.ylabel('x doubledot')
ax.set_xlabel('time')
ax.grid()

fig9 = plt.figure()
ax = fig9.add_subplot()
plt.plot(xdot.value,ydot.value)
ax.set_title('X and Y velocity')
plt.ylabel('y velocity')
ax.set_xlabel('x velocity')
ax.grid()

fig10 = plt.figure()
ax = fig10.add_subplot()
plt.plot(ts*470, x.value)
ax.set_title('X position')
plt.ylabel('x')
ax.set_xlabel('time')
ax.grid()'''

plt.show()

print('final y',y.value[-1]*86904)
print('final x', x.value[-1]*-8690)
print('final ydot',ydot.value[-1]*86904)
print('final xdot', xdot.value[-1]*-86904)
print('final ydoubledot',ydoubledot.value[-1]*86904)
print('final xdoubledot', xdoubledot.value[-1]*-86904)
print('final time', tf.value[0]*final_time)

pos_factor = 86904
pos_offset = 1738100
y_pos_list = [0]*len(x.value)
x_pos_list = [0]*len(x.value)
theta_list = [0]*len(x.value)
for i in range(len(x.value)):
    x_pos_list[i] = -x.value[i]*86904
    y_pos_list[i] = y.value[i]*86904+pos_offset
    theta_list[i] = 3*angle.value[i]*(360/(2*np.pi))

angle_1 = np.linspace(0, 2*np.pi, 100)
radius = 1738100
a = radius*np.cos(angle_1)
b = radius*np.sin(angle_1)

plt1 = plt.figure()
ax = plt1.add_subplot()
plt.plot(a,b)
plt.plot(x_pos_list,y_pos_list)
ax.set_title('Position')
plt.ylabel('y')
ax.set_xlabel('x')
ax.grid()
ax.set_aspect('equal')
plt.savefig('takeoff_alone.png', dpi=300)

plt2 = plt.figure()
ax = plt2.add_subplot()
plt.plot(470*ts,theta_list)
ax.set_title('Angle')
plt.ylabel('Angle / degrees')
ax.set_xlabel('time')
ax.grid()
plt.savefig('angle.png', dpi=300)

plt3 = plt.figure()
ax = plt3.add_subplot()
plt.plot(x_pos_list,y_pos_list)
ax.set_title('Position')
plt.ylabel('y')
ax.set_xlabel('x')
ax.grid()
ax.set_aspect('equal')
plt.savefig('takeoff_context.png', dpi=300)

plt.show()
