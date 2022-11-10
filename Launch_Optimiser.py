# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 19:04:27 2022

@author: Bobke
"""

import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
from gekko import *


class Solver:
    
    
    """Setting up the solver"""
    
    m = GEKKO()
    nt = 200 #number of timesteps
    m.time = np.linspace(0,1,nt) # scale 0-1 time with tf
    
    
    """Solver Options"""
    m.options.NODES = 2 #The number of collocation points between each timestep
    m.options.SOLVER = 3 #1 = APOPT, 3 = IPOPT 
    m.options.IMODE = 6  #Tells gekko the problem is an optimal control problem
    m.options.MAX_ITER = 20000 
    m.options.MV_TYPE = 0 #How the interpolation between Manipulated variables (MVs) is done
    #m.options.COLDSTART= 2 #Should help find bad constraint
    m.options.OTOL = 1e-3 #allows for margins of error in solution, default of 1e-6
    m.options.RTOL = 1e-3 #same as above
    m.options.DIAGLEVEL = 1
    
    
    """Defining Problem Parameters"""
    
    final_time = 470  #The max launch time of rocket, limited by fuel
    tf = m.FV(value=0,lb=0,ub=1) #FV is a fixed value variable
    tf.STATUS = 1 #STATUS = 1 means the optimzer can minimize the function
    
    
    """Constants"""
    
    """Constants have been defined twice, once as normal python variables,
    to be able to use in general arithmatics, and once as GEKKO constants,
    which was done to help trouble shoot the model when testing. '_py' was
    used to denote the duplicate of constants for normal python variable"""
    
    G_py = 6.674*10**(-11)  #Gravitational Constant
    M_py = 7.346*10**(22)   #Mass of the Moon in kg
    R0_py = 1738100         #Radius of the lunar surface in meters
    
    #Gekko variables of constants    
    G  =  m.Const(G_py, name='G')       
    M  =  m.Const(M_py, name='M')       
    R0 =  m.Const(R0_py, name='R0')     
    

    """Rocket Parameters"""
    Ft =  m.Const(15346, name='Ft')         #thrust force of engine in Newtons
    M0 =  m.Const(4821, name='M0')          #Wet Mass of rocket, in kg
    M_dot =  m.Const(5.053, name='M_dot')   #Mass flow rate of propellant, kg/s
    fuel_mass = 2376                        #Mass of fuel, kg
    mflow = 5.053/fuel_mass
    angle_doubledot_max = 5e-4             #max angular acceleration (just a guess for now) in radians per second squared
    
    
    """"Orbit Parameters """
    r_periapsis = 17703                                 #Final orbit periapsis from lunar surface, m
    r_apoapsis = 88615                                  #Final orbit apoapsis from lunar surface, m
    r_avg = (r_periapsis+r_apoapsis)/2
    Rfmin_py = r_periapsis                              #final height of launch, which will be the orbit's periapsis                      
    Rfmin = m.Const(Rfmin_py, name ='Rfmin')                
    periapsis_v = ((G_py*M_py)/(R0_py+r_avg))**(1/2)    #Velocity at end of launch
    
    """The periapsis velocity (periapsis_v) is calculated by finding the velocity
    the rocket would have in a circular orbit of radius (r_apoapsis+r_periapsis)/2. """
    
     

    """ Setting up Gekko Variables prone to changes"""
    mass = m.Var(value=0, lb = 0, ub = 1, name='mass') #lb, ub are upper and lower bounds respectively
    
    # Position variables
    y = m.Var(name='y', value = 0)
    ydot = m.Var( name='ydot')
    ydoubledot = m.Var(name='ydoubledot')
    
    x = m.Var(name='x', value = 0)
    xdot = m.Var( name='xdot')
    xdoubledot = m.Var(name='xdoubledot')
    
    angle = m.Var(name='angle', value = 0, lb = 0, ub = np.pi/3)
    angledot = m.Var(name='angledot')
    angledoubledot = m.MV(name='angledoubledot', lb = -1, ub = 1)
    #angledot.DPRED
    angledoubledot.STATUS = 1 #Allows computer to change angle
    angledoubledot.DCOST = 1e-5 #Adds a very small cost to changes in angle
    angledoubledot.REQONCTRL = 3 #tells solver whether to change MVs or run as simulator
    
    
    """"Scalars --it is crucial to have all the variables change at a similar scale to each other
    The final orbit height is used as the distance scale factor, the max fuel mass is used as 
    the mass scale factor"""
    
    Scalar = m.Const(Rfmin_py, name = 'distance Scale') 
    mass_scalar = m.Const(fuel_mass, name = 'mass Scale') #The fuel mass of the rocket
    angle_scalar = m.Const(angle_doubledot_max/3)
    
    
    """Governing Equations"""
    # differential equations scaled
    m.Equation(y.dt()==tf*ydot*final_time) #Expression for y velocity
    m.Equation(ydot.dt() == tf*ydoubledot*final_time) #Expression for y acceleration
    
    m.Equation(x.dt()==tf*xdot*final_time) #Expression for x velocity
    m.Equation(xdot.dt() == tf*xdoubledot*final_time) #Expression for x acceleration
    
    m.Equation(angle.dt() == tf*angledot*final_time) #Expression for angle
    m.Equation(angledot.dt() == tf*angledoubledot*final_time*angle_scalar)
    
    m.Equation(mass.dt() == mflow*final_time*tf) #Expression for mass
    
    
    """System Dynamics"""
    m.Equation(ydoubledot == ( ( (Ft/((M0-mass_scalar*mass)*((x*Scalar)**2 + (y*Scalar+R0)**2)**(1/2)))*\
                              ((y*Scalar+R0)*m.cos(3*angle)+(x*Scalar)*m.sin(3*angle)) )\
               - (y*Scalar+R0)*(G*M/(((x*Scalar)**2 + (y*Scalar+R0)**2)**(3/2))) )/ Scalar\
               )
    
    
    m.Equation(xdoubledot == ( ( (Ft/((M0-mass_scalar*mass)*((x*Scalar)**2 + (y*Scalar+R0)**2)**(1/2)))*\
                              ((x*Scalar)*m.cos(3*angle)-(y*Scalar+R0)*m.sin(3*angle)) )\
               - (x*Scalar)*(G*M/(((x*Scalar)**2 + (y*Scalar+R0)**2)**(3/2))))/ Scalar\
               )
        
    """For more information about the governing equations and system dynamics, 
    reffer to the Lunar_Module_Trajectory_Optimisation.pdf file attached on the github page.
    This document was written as a school assessment, so apologies for the unconventional formating"""
    
    
    
    """Initial Boundary Conditions:""" 
    m.fix(y, pos=0,val=0)       #Initial y position
    m.fix(x, pos=0,val=0)       #Initial x position
    m.fix(ydot, pos=0,val=0)    #Initial y velocity
    m.fix(xdot, pos=0,val=0)    #Initial x velocity
    
    m.fix(angle, pos=0, val=0)  #Initial angle
    m.fix(mass, pos=0,val=0) #Initial mass  
    
    
    """Final Boundary Conditions"""  
    
    """Constraint 1: Make sure final orbital radius is greater than Rfmin """     
    #Creates a list that satisfies final condition at all time except at final time for constraint_1
    constraint_1 = np.full(nt, Rfmin+R0+1)
    constraint_1 [-1] = 0
    final_radius = m.Param(value = constraint_1 )
    m.Equation(((y+R0/Scalar)**2+(x)**2)**(1/2)+final_radius >= ((R0+Scalar)/Scalar) )
    
    
    """Constraint 2: ensures final velocity is orbital"""  
    #Creates a list that satisfies final condition at all time except at final time for constraint_2
    constraint_2 = np.full(nt, 0)
    constraint_2 [-1] = 1
    final_velocity = m.Param(value = constraint_2 )
    m.Equation((xdot**2 + ydot**2) >= (periapsis_v/Scalar)**2*final_velocity)
    
    
    """Constraint 3: dot product of velocity and position should be zero in a circular orbit"""
    m.Equation(((y*Scalar+R0)*(ydot*Scalar) +(x*Scalar)*(xdot*Scalar))*final_velocity == 0)
    
    """Solving the Problem"""
    m.Minimize(tf) 
    m.solve(disp=True)    #solve
    print('Optimal Solution (final time): ' + str(tf.value[0]*470))
    print(periapsis_v)

    

    
    """Solution Display"""
    
    # scaled time
    ts = m.time * tf.value[0]
    print('final y',y.value[-1]*Rfmin_py)
    print('final x', x.value[-1]*Rfmin_py)
    print('final ydot',ydot.value[-1]*Rfmin_py)
    print('final xdot', xdot.value[-1]*Rfmin_py)
    print('final ydoubledot',ydoubledot.value[-1]*Rfmin_py)
    print('final xdoubledot', xdoubledot.value[-1]*Rfmin_py)
    print('final time', tf.value[0]*final_time)
    
    y_pos_list = [0]*len(x.value)
    x_pos_list = [0]*len(x.value)
    theta_list = [0]*len(x.value)
    for i in range(len(x.value)):
        x_pos_list[i] = -x.value[i]*Rfmin_py
        y_pos_list[i] = y.value[i]*Rfmin_py+R0_py
        theta_list[i] = 3*angle.value[i]*(180/(np.pi))
    
    angle_1 = np.linspace(0, np.pi/6, 100)
    a = R0_py*np.cos(angle_1)
    b = R0_py*np.sin(angle_1)
    
    circle1 = plt.Circle((0, 0), R0_py)
    
    fig, ax = plt.subplots()
    #plt.plot(a,b)
    ax.add_patch(circle1)
    plt.plot(x_pos_list,y_pos_list, color = (0.9, 0.4, 0))
    ax.set_ylim(R0_py-30000, R0_py+20000)
    ax.set_xlim(-5000, 300000) 
    
    ax.set_title('Position')
    plt.ylabel('y')
    ax.set_xlabel('x')
    ax.grid()
    ax.set_aspect('equal')
    plt.savefig('takeoff_contextualized.png', dpi=300)
    
    plt2 = plt.figure()
    ax = plt2.add_subplot()
    plt.plot(final_time*ts,theta_list)
    ax.set_title('Angle')
    plt.ylabel('Angle / degrees')
    ax.set_xlabel('time')
    ax.grid()
    plt.savefig('Angle_vs_Time.png', dpi=300)
    
    plt3 = plt.figure()
    ax = plt3.add_subplot()
    plt.plot(x_pos_list,y_pos_list)
    ax.set_title('Position')
    plt.ylim([R0_py-10000, R0_py+20000])
    plt.ylabel('y')
    ax.set_xlabel('x')
    ax.grid()
    ax.set_aspect('equal')
    plt.savefig('takeoff_trajectory.png', dpi=300)
    
    plt.show()

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


