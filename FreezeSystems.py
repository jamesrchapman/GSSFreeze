import numpy as np
import matplotlib.pyplot as plt

# Define parameters
k = 5  # example value, adjust as needed
r = 5  # example value, adjust as needed
g = 5.0  # example value, adjust as needed
c = 5.0  # example value, adjust as needed

# Define the vector field equations
def dF(F, T):
    return k * T - r * F

def dT(F, T):
    return np.where(F != 0, g - c / F, 0)  # Handle division by zero with np.where

# Set up a grid for F and T values
F_vals = np.linspace(0.1, 2.1, 20)  # Avoid zero for F to prevent division by zero
T_vals = np.linspace(0.1, 2.1, 20)
F, T = np.meshgrid(F_vals, T_vals)

# Compute vector field
dF_vals = dF(F, T)
dT_vals = dT(F, T)

# Calculate equilibrium point
F_eq = c / g
T_eq = r * c / (k * g)

# Plot the vector field and equilibrium point
plt.figure(figsize=(10, 8))
plt.quiver(F, T, dF_vals, dT_vals, color='b', alpha=0.6)
plt.plot(F_eq, T_eq, 'ro', label="Equilibrium Point")
plt.xlabel('F')
plt.ylabel('T')
plt.title('Vector Field of the Dynamical System with Equilibrium Point')
plt.legend()
plt.grid()
plt.show()
