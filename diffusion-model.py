#!/usr/bin/env python
# coding: utf-8

# # A 1D diffusion model
# 
# 

# Here we develope a one dimensional model of diffusion. it assumes a constant diffusivity. it uses a regular grid. it has a step function for an initial condition. it has fixed boundary
# condition.    here we develope a one dimensional model of diffusion. it assumes a constant diffusivity. it uses a regular grid. it has a step function for an initial condition. it has fixed boundary condition.
# 

# here is the diffusion equation.
# 

# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$
# 
# 

# Here is the discretized version of the diffusion equation we will solve with our model:

# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2}
# (C^t_{x+1} - 2C^t_x + C^t_{
#     x-1}) $$

# This is the FTCS scheme as described by Slingerland and kump (2011)

# we'll use two libraries, NumPy (for arrays) and Matplotlib (for plotting), that aren't a part of the core python distribution.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt 


# Start by setting two fixed model parameters, the diffusivity and the size of the model domain. 

# In[ ]:


D = 100
Lx = 300


# Next, set up the model grid using a NumPy array.

# In[ ]:


dx = 0.5
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x)


# In[ ]:


nx


# In[ ]:


x[0:5]


# In[ ]:


x[-5:-1]


# In[ ]:


x[-5:]


# Set the initial conditions for the model.
# The cake `C` is a step function with a high value of the left, a low value on the right, and a step at the center fo the demain.

# In[ ]:


C = np.zeros_like(x)
C_left = 500
C_right = 0
C[x<=Lx/2]=C_left
C[x>Lx/2]=C_right


# Plot the initial profile.

# In[ ]:


plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial profile")


# Set the number of time steps in the model. Calculate a stable time step using a stability criterian. 

# In[ ]:


nt = 5000
dt = 0.5 * dx ** 2 / D


# Loop over the time steps of the model, solving the diffusion equation using the FTCS scheme shown above.
# Note the use of array operations on the variable `C`.
# The boundary conditions remain fixed in each time step.

# In[ ]:


for t in range(0, nt):
     C[1:-1] += D * dt / dx ** 2 * (C[:-2] - 2*C[1:-1] + C[2:])


# In[ ]:


z = list(range(5))
z


# In[ ]:


z[1:-1]


# In[ ]:


z[:-2]


# In[ ]:


z[2:]


# Plot the result

# In[ ]:


plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial profile")

