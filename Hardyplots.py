import numpy as np
import matplotlib.pyplot as plt

# Hardy-coëfficiënten 

g = np.array([
    -2.8365744e3,   
    -6.028076559e3, 
    1.954263612e1,  
    -2.737830188e-2,
    1.6261698e-5,   
    7.0229056e-10,  
    -1.8680009e-13, 
    2.7150305       
])

k = np.array([
    -5.8666426e3,
    2.232870244e1,
    1.39387003e-2,  
    -3.4262402e-5,  
    2.7040955e-8,   
    6.7063522e-1    
])

# Functies voor verzadigingsdampdruk

def e_w(T_K):
    """
    Verzadigingsdampdruk boven water [Pa]
    volgens Hardy-formule, T in Kelvin.
    """
    power_terms = sum(g[i] * T_K**(i-2) for i in range(7))
    ln_term = g[7] * np.log(T_K)
    return np.exp(power_terms + ln_term)

def e_i(T_K):
    """
    Verzadigingsdampdruk boven ijs [Pa]
    volgens Hardy-formule, T in Kelvin.
    """
    power_terms = sum(k[i] * T_K**(i-1) for i in range(5))  
    ln_term = k[5] * np.log(T_K)
    return np.exp(power_terms + ln_term)

# Temperatuur-ranges (in °C)

T_C_water = np.linspace(0, 100, 500)    
T_K_water = T_C_water + 273.15

T_C_ice = np.linspace(-80, 0, 500)      
T_K_ice = T_C_ice + 273.15

# Dampdrukken berekenen

e_water = e_w(T_K_water)  # [Pa]
e_ice = e_i(T_K_ice)      # [Pa]

plt.figure()
plt.plot(T_C_water, e_water)
plt.xlabel("$T$ [°C]")
plt.ylabel("e$_s$ [Pa]")
plt.grid()
plt.show()

plt.figure()
plt.plot(T_C_ice, e_ice)
plt.xlabel("$T$ [°C]")
plt.ylabel("e$_s$ [Pa]")
plt.grid()
plt.show()
