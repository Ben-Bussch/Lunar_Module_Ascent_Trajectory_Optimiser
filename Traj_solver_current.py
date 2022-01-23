# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 17:13:30 2022

@author: Bobke
"""

from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.environ import Set, ConcreteModel, TransformationFactory, Var, \
                          Reals, NonNegativeReals, PositiveReals, Constraint, \
                          SolverFactory, Objective, cos, sin, minimize, \
                          NonNegativeReals, ConstraintList
import numpy as np
import matplotlib.pyplot as plt

# Parameters of the problem
G = 6.674*10**(-11) #Gravitational Constant
M = 7.346*10**(22)  #Mass of the Moon
R0 = 1738100        #Radius of the lunar surface

Ft = 15346          #thrust force of engine
M0 = 4821           #Wet Mass of rocket
M_dot = 5.053       #Mass flow rate of propellant

Rfmin = 86904.6+R0  #Minimum final orbital radius
Rfmax = 106217+R0   #Maximum final orbital radius


#State Variables
model = ConcreteModel("rocket")
model.T = Var(domain=NonNegativeReals)
model.t = ContinuousSet(bounds=(0, 1))


model.x = Var(model.t, domain=Reals)
model.y = Var(model.t, domain=Reals)

model.xdot = DerivativeVar(model.x, wrt=model.t, domain=Reals)
model.xdoubledot = DerivativeVar(model.xdot, wrt=model.t)
model.ydot = DerivativeVar(model.y, wrt=model.t, domain=Reals)
model.ydoubledot = DerivativeVar(model.ydot, wrt=model.t)
model.pheta = Var(model.t, bounds=(-np.pi,np.pi))


#Making sure the lander does not colide with the moon
model.collide = Constraint(model.t, rule=lambda model, t:\
    ((model.x[t])**2 + (model.y[t])**2) >= R0**2\
        ) 

    
# System Dynamics
model.xode = Constraint(model.t, rule=lambda model, t: model.xdoubledot[t]+1e-9/(model.T**2+1e-9) == \
                        ( (7*((model.x[t]**2 + model.y[t]**2+1)**(1/2))) \
                            *(model.x[t]*cos(model.pheta[t]) - model.y[t]*sin(model.pheta[t])))\
                              - (model.x[t]*((G*M)/((model.x[t]**2 + model.y[t]**2+1)**(3/2))))\
                                  )

model.yode = Constraint(model.t, rule=lambda model, t: model.ydoubledot[t]+1e-9/(model.T**2+1e-9) == \
                        ( (7*((model.x[t]**2 + model.y[t]**2+1)**(1/2))) \
                         *(model.y[t]*cos(model.pheta[t]) + model.x[t]*sin(model.pheta[t])))\
                            - (model.y[t]*((G*M)/((model.x[t]**2 + model.y[t]**2+1)**(3/2))))\
                                )
      
    
# Initial boundary conditions 
model.x[0].fix(1)
model.y[0].fix(1738200)
model.xdot[0].fix(0)
model.ydot[0].fix(0)



# Final Boundary Condtitions 

#Initialize a guess for final x and y to avoid 
model.x[1] = 30000
model.y[1] = R0+80000
model.xdot[1] = 1400
model.ydot[1] = -200

#Minimum final orbit radius
model.r = Constraint(model.t, rule=lambda model, t: Constraint.Skip if t != model.t.last() else \
    (model.x[t]**2+ model.y[t]**2) >= Rfmin**2\
        )
   

#Direction of orbital velocity using dot product
model.velocity_direction_min = Constraint(model.t, rule=lambda model, t: Constraint.Skip if t != model.t.last() else \
                                model.x[t]*model.xdot[t]*model.T + model.y[t]*model.ydot[t]*model.T == 0)

  
#Minimum final orbital speed    
model.v_final = Constraint(model.t, rule=lambda model, t: Constraint.Skip if t != model.t.last() else \
                     (model.xdot[t]**2 + model.ydot[t]**2)**(1/2) >= ((G*M)/((model.x[t]**2+model.y[t]**2+1e-5)**(1/2)))\
                        )
  


    
    
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(model, wrt=model.t, nfe=30, ncp=6)

discretizer.reduce_collocation_points(model, var=model.pheta, ncp=1, contset=model.t)

model.obj = Objective(expr=model.T, sense=minimize)



solver = SolverFactory('ipopt')
solver.options["halt_on_ampl_error"] = "yes" 
results = solver.solve(model, tee=True)

tf_direct = model.T()
tdirect = [t*tf_direct for t in model.t]
phetadirect = [model.pheta[t]() for t in model.t]
xdirect = [model.x[t]() for t in model.t]
ydirect = [model.y[t]() for t in model.t]
xdotdirect = [model.xdot[t]()/tf_direct for t in model.t]
ydotdirect = [model.ydot[t]()/tf_direct for t in model.t]
xdoubledotdirect = [model.xdoubledot[t]() for t in model.t]
ydoubledotdirect = [model.ydoubledot[t]() for t in model.t]



fig1 = plt.figure()

ax = fig1.add_subplot()
ax.plot(tdirect, xdirect)
ax.set_title('x, m')
ax.set_xlabel('Time, sec')
ax.grid()

fig2 = plt.figure()
ax = fig2.add_subplot()
ax.plot(tdirect, ydirect)
ax.set_title('y, m')
ax.set_xlabel('Time, sec')
ax.grid()

fig3 = plt.figure()
ax = fig3.add_subplot()
ax.plot(tdirect, xdotdirect)
ax.set_title(r'$V_x$, m/s')
ax.set_xlabel('Time, sec')
ax.grid()

fig4 = plt.figure()
ax = fig4.add_subplot()
ax.plot(tdirect, ydotdirect)
ax.set_title(r'$V_y$, m/s')
ax.set_xlabel('Time, sec')
ax.grid()

fig5 = plt.figure()
ax = fig5.add_subplot()
ax.plot(tdirect, phetadirect)
ax.set_title('pheta')
ax.set_xlabel('Time, sec')
ax.grid()


angle = np.linspace(0, 2*np.pi, 100)
radius = 1738100
a = radius*np.cos(angle)
b = radius*np.sin(angle)

fig6 = plt.figure()
ax = fig6.add_subplot()
ax.plot(xdirect, ydirect)
ax.plot(a, b)
ax.set_title('Position')
ax.grid()

plt.show()

'''Error in an AMPL evaluation. Run with "halt_on_ampl_error yes" to see details.
Error evaluating Jacobian of equality constraints at user provided starting point.
  No scaling factors for equality constraints computed!
Error in an AMPL evaluation. Run with "halt_on_ampl_error yes" to see details.
Error evaluating Jacobian of inequality constraints at user provided starting point.'''
