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

#Constants:
G =  m.Const(6.674*10**(-11), name='G') #Gravitational Constant
M =  m.Const(7.346*10**(22), name='M')  #Mass of the Moon


#User Tunable
R0 =  m.Const(1738100, name='R0')       #Radius of the lunar surface
Ft =  m.Const(15346, name='Ft')          #thrust force of engine
M0 =  m.Const(4821, name='M0')            #Wet Mass of rocket
M_dot =  m.Const(5.053, name='M_dot')       #Mass flow rate of propellant
Rfmin = m.Const(53108.4, name ='Rfmin') #final height of orbit



orbital_v = ((6.674*10**(-11)*7.346*10**(22))/(1738100+53108.4))**(1/2)
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
Scalar = m.Const(53108.4, name = 'distance Scale')
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
    

#Final Boundary Conditions

#Creates a list that satisfies final condition at all time except at final time for constraint_1
constraint_1 = np.full(nt, Rfmin+R0+1)
constraint_1 [-1] = 0
final_radius = m.Param(value = constraint_1 )

m.Equation(((y+R0/Scalar)**2+(x)**2)**(1/2)+final_radius >= ((R0+Scalar)/Scalar))#ensures final radius is greater or equal to rfmin

#Creates a list that satisfies final condition at all time except at final time for constraint_2
constraint_2 = np.full(nt, 0)
constraint_2 [-1] = 1
final_velocity = m.Param(value = constraint_2 )

m.Equation((xdot**2 + ydot**2) >= (orbital_v/Scalar)**2*final_velocity)#ensures final velocity is orbital

# dot product of velocity and position should be zero
m.Equation(((y*Scalar+R0)*(ydot*Scalar)+(x*Scalar)*(xdot*Scalar))*final_velocity == 0)

m.Minimize(tf)

#Solving the problem
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
pos_factor = 53108.4
pos_offset = 1738100

print('final y',y.value[-1]*pos_factor)
print('final x', x.value[-1]*-pos_factor)
print('final ydot',ydot.value[-1]*pos_factor)
print('final xdot', xdot.value[-1]*-pos_factor)
print('final ydoubledot',ydoubledot.value[-1]*pos_factor)
print('final xdoubledot', xdoubledot.value[-1]*-pos_factor)
print('final time', tf.value[0]*final_time)

print('')
print('x5', angle.value[10]*3)


#SECOND PART OF THE CODE: produces graphs and computes the orbit with the trajectory information

y_pos_list = [0]*len(x.value)
x_pos_list = [0]*len(x.value)
theta_list = [0]*len(x.value)
y_v_list = np.zeros(len(x.value))
x_v_list = np.zeros(len(x.value))
y_a_list = np.zeros(len(x.value))
x_a_list = np.zeros(len(x.value))

for i in range(len(x.value)):
    x_pos_list[i] = -x.value[i]*pos_factor
    y_pos_list[i] = y.value[i]*pos_factor+pos_offset
    y_v_list[i] = ydot.value[i]*pos_factor
    x_v_list[i] = -xdot.value[i]*pos_factor
    y_a_list[i] = ydoubledot.value[i]*pos_factor
    x_a_list[i] = -xdoubledot.value[i]*pos_factor
    theta_list[i] = 3*angle.value[i]*(360/(2*np.pi))

#Constants
Gs = 6.67*10**-11
m_1 = 2000
m_2 = 7.346*10**(22)
f_y = y.value[-1]*pos_factor+pos_offset
f_x = x.value[-1]*-pos_factor
f_ydot = ydot.value[-1]*pos_factor
f_xdot = xdot.value[-1]*-pos_factor

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
    
    delta_t = 0.001 #time step
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

test = position(6600)

theta = np.linspace(0, 2*np.pi, 100)
a = radius*np.cos(theta)
b = radius*np.sin(theta)

pos_1_x_array = np.array(test[1])
pos_1_y_array = np.array(test[2])

#plt.style.use('ggplot')
x_pos = np.concatenate((x_pos_list, pos_1_x_array))    
y_pos = np.concatenate((y_pos_list, pos_1_y_array)) 


plt1 = plt.figure()
ax = plt1.add_subplot()
plt.plot(a, b)
plt.plot(x_pos , y_pos)
plt.title("LM Trajectory to Lunar Orbit")
plt.xlabel("x / meters")
plt.ylabel("y / meters")
ax.set_aspect(1,adjustable='datalim')
ax.grid()
plt.savefig('assembled.png', dpi=500)


plt2 = plt.figure()
ax = plt2.add_subplot()
plt.plot(470*ts,theta_list)
ax.set_title('Angle')
plt.ylabel('Angle / degrees')
ax.set_xlabel('time / seconds')
ax.grid()
plt.savefig('angle.png', dpi=300)

plt3 = plt.figure()
ax = plt3.add_subplot()
plt.plot(x_pos_list,y_pos_list)
ax.set_title('LM Trajectory Path')
plt.ylabel('y / meters')
ax.set_xlabel('x / meters')
ax.set(xlim=(0, 300000), ylim=(pos_offset, pos_offset+50000))
ax.set_aspect('equal')
ax.grid()
plt.savefig('takeoff_context.png', dpi=300)

plt4 = plt.figure()
ax = plt4.add_subplot()
plt.plot(x_v_list,y_v_list)
ax.set_title('LM X and Y Velocity')
plt.ylabel('y velocity / ms-1')
ax.set_xlabel('x velocity / ms-1')
ax.set_aspect('equal')
ax.grid()
plt.savefig('takeoff_velocity.png', dpi=500)

plt5 = plt.figure()
ax = plt5.add_subplot()
plt.plot(x_a_list,y_a_list)
ax.set_title('LM X and Y Acceleration')
plt.ylabel('y acceleration / ms-2')
ax.set_xlabel('x acceleration / ms-2')
ax.set_aspect('equal')
ax.grid()
plt.savefig('takeoff_acceleration.png', dpi=500)

plt.show()
