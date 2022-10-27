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
R0 =  1738100       #Radius of the lunar surface

Ft =  m.Const(15346, name='Ft')          #thrust force of engine
M0 =  m.Const(4821, name='M0')          #Wet Mass of rocket

orbital_v = ((6.674*10**(-11)*7.346*10**(22))/(1738100+106000))**(1/2)

# final time
final_time = 470 #470
tf = m.FV(value=0,lb=0,ub=1) #FV is a fixed value variable
tf.STATUS = 1 #STATUS = 1 means the optimzer can minimize the function

#Mass
mflow = 5.053/2376
mass = m.Var(value=0, lb = 0, ub = 1, name='mass') #lb, ub are upper and lower bounds respectively

# Position variables
y = m.Var(name='y', value = 0)
ydot = m.Var( name='ydot')
ydoubledot = m.Var(name='ydoubledot')


#Distance scale:
Scalar = m.Const(86904, name = 'Scale')
mass_scalar = m.Const(2576, name = 'mass Scale')


x = m.Var(name='x', value = 0)
xdot = m.Var( name='xdot')
xdoubledot = m.Var(name='xdoubledot')


angle = m.MV(name='angle', value = 0, lb = 0, ub = (np.pi)/3)
#angle.DPRED
angle.STATUS = 1 #Allows computer to change theta
angle.DCOST = 1e-5 #Adds a very small cost to changes in theta
angle.REQONCTRL = 3 #tells solver whether to change MVs or run as simulator

# differential equations scaled by tf
m.Equation(y.dt()==tf*ydot*final_time) #Expression for y velocity
m.Equation(ydot.dt() == tf*ydoubledot*final_time) #Expression for y acceleration

m.Equation(x.dt()==tf*xdot*final_time) #Expression for x velocity
m.Equation(xdot.dt() == tf*xdoubledot*final_time) #Expression for x acceleration

m.Equation(mass.dt() == mflow*final_time*tf) #Expression for mass

#system dynamics
m.Equation(ydoubledot == ((Ft/(M0-mass_scalar*mass))*m.cos(6*angle)-G*M/(y*Scalar+R0)**2)/Scalar)
m.Equation(xdoubledot == ((Ft/(M0-mass_scalar*mass))*m.sin(6*angle)/Scalar))

#Creates a list that satisfies final condition at all time except at final time for constraint_1
constraint_1 = np.full(nt, 10)
constraint_1 [-1] = 0

final_radius = m.Param(value = constraint_1 )
#Make sure final orbital radius is greater than Rfmin
m.Equation(y+final_radius >= 0.8)
m.Equation(xdot+final_radius > orbital_v/Scalar)
#m.Equation(xdot+final_radius > orbital_v/Scalar)

#Creates a list that satisfies final condition at all time except at final time for constraint_2
constraint_2 = np.full(nt, -500)
constraint_2 [-1] = 1
final_velocity = m.Param(value = constraint_2 )


m.Equation(ydot*final_velocity <= -19/(Scalar-2000)) #Enbsures minimal vertical component of velocity


#m.Maximize(xdot)
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
plt.style.use('ggplot')

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
ax.grid()'''

print('final y',y.value[-1]*86904)
print('final x', x.value[-1]*8690)
print('final ydot',ydot.value[-1]*86904)
print('final xdot', xdot.value[-1]*86904)
print('final ydoubledot',ydoubledot.value[-1]*86904)
print('final xdoubledot', xdoubledot.value[-1]*86904)
print('final time', tf.value[0]*final_time)

#Constants
Gs = 6.67*10**-11
m_1 = 2000
m_2 = 7.346*10**(22)

f_y = y.value[-1]*86904+R0
f_x = x.value[-1]*86904
f_ydot = ydot.value[-1]*86904
f_xdot = xdot.value[-1]*86904

#Initial Conditions
radius = 1738100

position_1_0 = [f_x, f_y] #on surface
position_2_0 = [0,0] #yeeted out
position_1_dot_0 = [f_xdot, f_ydot]
position_2_dot_0 = [0,0]
#Defenition of ODE
def get_position_1_double_dot(pos_1, pos_2):
    distance =np.sqrt((pos_2[0] - pos_1[0])**2+(pos_2[1] - pos_1[1])**2)
    #print(distance) 
    x_1_direction = pos_2[0] - pos_1[0]
    y_1_direction = pos_2[1] - pos_1[1]
    x_1_double_dot = Gs*m_2*(x_1_direction/distance)*(1/(distance**2))
    y_1_double_dot = Gs*m_2*(y_1_direction/distance)*(1/(distance**2))
    #if distance < 1737400:
        #print("ground hit")
    
    return [x_1_double_dot,y_1_double_dot]

#Solution to DE
def position(t):
    pos_1_x_list = []
    pos_1_y_list = []
    pos_1_double_dot_list = []
    
    position_1 = position_1_0
    position_2 = position_2_0
    position_1_dot = position_1_dot_0
    position_2_dot = position_2_dot_0
    
    delta_t = 0.005 #time step
    time_list = np.arange(0,t,delta_t)
    for time in np.arange(0,t,delta_t):
        position_1_double_dot = get_position_1_double_dot(position_1,position_2)
        pos_1_double_dot_list.append(position_1_double_dot)
        
        position_1[0] += position_1_dot[0]*delta_t
        position_1[1] += position_1_dot[1]*delta_t
        position_1_dot[0] += position_1_double_dot[0]*delta_t
        position_1_dot[1] += position_1_double_dot[1]*delta_t
        pos_1_x_list.append(position_1[0])
        pos_1_y_list.append(position_1[1])
    return [time_list, pos_1_x_list, pos_1_y_list]

test = position(5000)


# scaled time
ts = m.time * tf.value[0]

theta = np.linspace(0, 2*np.pi, 100)

a = radius*np.cos(theta)
b = radius*np.sin(theta)


pos_1_x_array = np.array(test[1])
#print(pos_1_x_array)

pos_1_y_array = np.array(test[2])
#print(pos_1_y_array)

#plt.style.use('ggplot')


pos_factor = 86904
pos_offset = 1738100
y_pos_list = np.zeros(len(x.value))
x_pos_list = np.zeros(len(x.value))
theta_list = [0]*len(x.value)
for i in range(len(x.value)):
    x_pos_list[i] = x.value[i]*86904
    y_pos_list[i] = y.value[i]*86904+pos_offset
    theta_list[i] = 6*angle.value[i]*(360/(2*np.pi))


x_pos = np.concatenate((x_pos_list, pos_1_x_array))    
y_pos = np.concatenate((y_pos_list, pos_1_y_array)) 
plt.figure(num = 0, dpi = 1000)
plt.plot(a, b)
plt.plot(x_pos , y_pos)
plt.title("coordinates")
plt.xlabel("x")
plt.ylabel("y")
ax = plt.gca()
ax.set_aspect(1,adjustable='datalim')

plt.savefig('assembled.png', dpi=300)


plt2 = plt.figure()
ax = plt2.add_subplot()
plt.plot(470*ts,theta_list)
ax.set_title('Angle')
plt.ylabel('Angle / degrees')
ax.set_xlabel('time')
ax.grid()
plt.savefig('angle.png', dpi=300)


plt.show()