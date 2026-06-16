### T2rho.py
#
# Given a desired Temp, calculate density for input file

import numpy as np


## Air
T = 600;
P = 1e5;
gamma = 1.4;
P0 = 0.0;
Cv = 717.86;


rho = (P + gamma*P0) / (T*Cv*(gamma-1.0));
print("Air Density:"+str(rho));

### Water
T = 373;
P = 1e5;
gamma = 7.15;
P0 = 3.0e8;
Cv = 584.47;

rho = (P + gamma*P0) / (T*Cv*(gamma-1.0));
print("Water Density:"+str(rho));

