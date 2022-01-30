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

'''IPOPT results after 1000 iterations:
    Number of Iterations....: 1000

                                       (scaled)                 (unscaled)
    Objective...............:   7.9813943414888072e+08    7.1832549073399267e+09
    Dual infeasibility......:   3.9191396300710375e+13    3.5272256670639338e+14
    Constraint violation....:   5.4864024536848068e+04    3.1566804735840803e+08
    Complementarity.........:   1.0000000000000002e+13    9.0000000000000016e+13
    Overall NLP error.......:   5.4864024536848068e+04    3.5272256670639338e+14


    Number of objective function evaluations             = 12580
    Number of objective gradient evaluations             = 924
    Number of equality constraint evaluations            = 12586
    Number of inequality constraint evaluations          = 12586
    Number of equality constraint Jacobian evaluations   = 1023
    Number of inequality constraint Jacobian evaluations = 1023
    Number of Lagrangian Hessian evaluations             = 1000
    Total CPU secs in IPOPT (w/o function evaluations)   =  11277.210
    Total CPU secs in NLP function evaluations           =    578.209

    EXIT: Maximum Number of Iterations Exceeded.
     
     An error occured.
     The error code is           -1
     
     
     ---------------------------------------------------
     Solver         :  IPOPT (v3.12)
     Solution time  :    11893.4746000000      sec
     Objective      :    7183254907.33993     
     Unsuccessful with error code            0
     ---------------------------------------------------
     
     Creating file: infeasibilities.txt
     Use command apm_get(server,app,'infeasibilities.txt') to retrieve file
     Called files(          21 )
     Called files(           2 )
     Called files(          53 )
     WRITE dbs FILE
     Called files(          56 )
     WRITE json FILE
     Called files(           2 )
     Called files(           3 )
     Called files(          21 )
     Called files(          23 )
     Called files(          11 )
     Files(11): File Read warm.t0 F
     files: warm.t0 does not exist
     Called files(          12 )
     Files(12): File Read lam.t0 F
     files: lam.t0 does not exist
    Timer #     1   11898.92/       1 =   11898.92 Total system time
    Timer #     2   11893.47/       1 =   11893.47 Total solve time
    Timer #     3      59.93/   12580 =       0.00 Objective Calc: apm_p
    Timer #     4       6.94/     925 =       0.01 Objective Grad: apm_g
    Timer #     5      57.34/   12586 =       0.00 Constraint Calc: apm_c
    Timer #     6       0.00/       1 =       0.00 Sparsity: apm_s
    Timer #     7       5.05/    1024 =       0.00 1st Deriv #1: apm_a1
    Timer #     8       0.00/       0 =       0.00 1st Deriv #2: apm_a2
    Timer #     9       0.24/    1200 =       0.00 Custom Init: apm_custom_init
    Timer #    10       0.01/    1200 =       0.00 Mode: apm_node_res::case 0
    Timer #    11       0.01/    3600 =       0.00 Mode: apm_node_res::case 1
    Timer #    12       0.12/    1200 =       0.00 Mode: apm_node_res::case 2
    Timer #    13       0.00/    2400 =       0.00 Mode: apm_node_res::case 3
    Timer #    14     282.37/30206400 =       0.00 Mode: apm_node_res::case 4
    Timer #    15      70.72/ 2338800 =       0.00 Mode: apm_node_res::case 5
    Timer #    16      80.19/ 1176000 =       0.00 Mode: apm_node_res::case 6
    Timer #    17       3.91/    1025 =       0.00 Base 1st Deriv: apm_jacobian
    Timer #    18       0.00/       0 =       0.00 Base 1st Deriv: apm_condensed_jacobian
    Timer #    19       0.01/       2 =       0.01 Non-zeros: apm_nnz
    Timer #    20       0.00/       0 =       0.00 Count: Division by zero
    Timer #    21       0.00/       0 =       0.00 Count: Argument of LOG10 negative
    Timer #    22       0.00/       0 =       0.00 Count: Argument of LOG negative
    Timer #    23       0.00/       0 =       0.00 Count: Argument of SQRT negative
    Timer #    24       0.00/       0 =       0.00 Count: Argument of ASIN illegal
    Timer #    25       0.00/       0 =       0.00 Count: Argument of ACOS illegal
    Timer #    26       0.00/       1 =       0.00 Extract sparsity: apm_sparsity
    Timer #    27       0.01/      32 =       0.00 Variable ordering: apm_var_order
    Timer #    28       0.00/       0 =       0.00 Condensed sparsity
    Timer #    29       0.73/       2 =       0.36 Hessian Non-zeros
    Timer #    30       0.04/       7 =       0.01 Differentials
    Timer #    31       5.01/     978 =       0.01 Hessian Calculation
    Timer #    32       3.03/     981 =       0.00 Extract Hessian
    Timer #    33       0.02/       2 =       0.01 Base 1st Deriv: apm_jac_order
    Timer #    34       0.01/       1 =       0.01 Solver Setup
    Timer #    35   11311.84/       1 =   11311.84 Solver Solution
    Timer #    36       6.87/   12600 =       0.00 Number of Variables
    Timer #    37       0.01/       8 =       0.00 Number of Equations
    Timer #    38       4.14/      31 =       0.13 File Read/Write
    Timer #    39       0.01/       1 =       0.01 Dynamic Init A
    Timer #    40       0.48/       1 =       0.48 Dynamic Init B
    Timer #    41       0.14/       1 =       0.14 Dynamic Init C
    Timer #    42       0.01/       1 =       0.01 Init: Read APM File
    Timer #    43       0.00/       1 =       0.00 Init: Parse Constants
    Timer #    44       0.00/       1 =       0.00 Init: Model Sizing
    Timer #    45       0.00/       1 =       0.00 Init: Allocate Memory
    Timer #    46       0.00/       1 =       0.00 Init: Parse Model
    Timer #    47       0.00/       1 =       0.00 Init: Check for Duplicates
    Timer #    48       0.00/       1 =       0.00 Init: Compile Equations
    Timer #    49       0.00/       1 =       0.00 Init: Check Uninitialized
    Timer #    50      -0.00/    1223 =      -0.00 Evaluate Expression Once
    Timer #    51       0.00/       0 =       0.00 Sensitivity Analysis: LU Factorization
    Timer #    52       0.00/       0 =       0.00 Sensitivity Analysis: Gauss Elimination
    Timer #    53       0.00/       0 =       0.00 Sensitivity Analysis: Total Time
     @error: Solution Not Found
    Not successful
    http://byu.apmonitor.com
    gk_model0'''