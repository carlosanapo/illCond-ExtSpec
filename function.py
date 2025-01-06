#%%
# Calcalute the spherical bessel function of the first kind j_n(x) and the hankel function h_n(x) and their derivatives
from scipy.special import spherical_jn, spherical_yn
import numpy as np
import matplotlib.pyplot as plt
# latex rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#%%
radius = np.linspace (100e-9,1000e-9,9)
radius = 700e-9
wavelength = 314e-9
x = 2 * np.pi * radius / wavelength
n_max = x + 4 * x**(1/3)
m = 1.33

n_range = np.arange(0, n_max , 1).astype(int)

jn_array = []
jn_prime_array = []
yn_array = []
yn_prime_array = []


plt.figure()    
for n in n_range:
    jn = spherical_jn(n, x)
    jn_prime = spherical_jn(n, x, derivative=True)
    jn_array.append(jn)
    jn_prime_array.append(jn_prime)
    
    yn = spherical_yn(n, x)
    yn_prime = spherical_yn(n, x, derivative=True)
    yn_array.append(yn)
    yn_prime_array.append(yn_prime)

# calculate incremental sum of jn
sum_jn_array = np.cumsum(jn_array)
sum_jn_prime_array = np.cumsum(jn_prime_array)

x = np.arange(0, len(jn_array), 1)

plt.plot(jn_array, label=f"n={n}")
plt.plot(x, sum_jn_array, label=f"sum_jn")
# plt.plot(x, sum_jn_prime_array, label=f"sum_jn_prime")
# plt.plot(yn_array, label=f"n={n}")
# plt.plot(yn_prime_array, label=f"n={n}")

plt.legend()
#%%
def jn_bessel(x, m):
    n_max = x + 4 * x**(1/3)
    n_range = np.arange(1, n_max , 1).astype(int)
    for n in n_range:
        jn = spherical_jn(n, x)
        jn_prime = spherical_jn(n, x, derivative=True)
        jn = np.sum(jn)
        jn_prime = np.sum(jn_prime)

    return jn, jn_prime

def yn_bessel(x, m):
    n_max = x + 4 * x**(1/3)
    n_range = np.arange(1, n_max , 1).astype(int)
    for n in n_range:
        yn = spherical_yn(n, x)
        yn_prime = spherical_yn(n, x, derivative=True)
        yn = np.sum(yn)
        yn_prime = np.sum(yn_prime)

    return yn, yn_prime

x = 2 * np.pi * radius / wavelength
m = 1.33

wavelength = 314e-9
radius = np.linspace (100e-9,5000e-9,1000)
x = 2 * np.pi * radius / wavelength

jarray = []
jprimearray = []

for i in range(len(x)):
    j, j_prime = jn_bessel(x[i], m)
    jarray.append(j)
    jprimearray.append(j_prime)

framewidth = 1.5  
plt.rcParams['axes.linewidth'] = framewidth
# latex rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(jarray,'-', linewidth=2, markersize=1, label='j', color='black')
plt.plot(jprimearray,'-', linewidth=2, markersize=1, label='j_prime', color='red')

# plt.legend(fancybox=False, shadow=False, loc='best', fontsize=12, edgecolor='black')
legend = plt.legend()
frame = legend.get_frame()
frame.set_linewidth(framewidth)
frame.set_edgecolor('black')
# change the legend box edge width
plt.xlabel(r'$x = 2 \pi r/ \lambda  \ [-]$',fontsize=12)
plt.ylabel(r'$j_n(mx)$',fontsize=12)
plt.title('Spherical Bessel function of the first kind', fontsize=12)
plt.xticks(fontsize=12), plt.yticks(fontsize=12)
plt.grid('True', linestyle='--', linewidth=0.5)
plt.xlim(-1.2, 50)
plt.ylim(-.000026, 0.0007)
# change width of all the plot borders

#%% SCATTERING COEFFICIENTS
# permeability of the medium (vacuum)
def jn_bessel(x):
    n_max = x + 4 * x**(1/3)
    n_range = np.arange(1, n_max , 1).astype(int)
    for n in n_range:
        jn = spherical_jn(n, x)
        jn_prime = spherical_jn(n, x, derivative=True)
        jn = np.sum(jn)
        jn_prime = np.sum(jn_prime)
    
    return jn, jn_prime

def yn_bessel(x):
    n_max = x + 4 * x**(1/3)
    n_range = np.arange(1, n_max , 1).astype(int)
    for n in n_range:
        yn = spherical_yn(n, x)
        yn_prime = spherical_yn(n, x, derivative=True)
        yn = np.sum(yn)
        yn_prime = np.sum(yn_prime)

    return yn, yn_prime

def n_array(x):
    narray = np.zeros(len(x))
    for i in range(len(x)):
        nmax = x[i] + 4 * x[i]**(1/3)
        n_range = np.arange(1, nmax , 1).astype(int)
        n_range = np.sum(2 * n_range + 1)
        narray[i] = n_range

    return narray

radius = np.linspace (400e-9,5000e-9,400)
wavelength = 500e-9
m = 1.33
x = 2 * np.pi * radius / wavelength
n_array = n_array(x)
a_n_array = np.zeros(len(x))
b_n_array = np.zeros(len(x))
for i in range(len(x)):
    jn_x, jn_prime_x = jn_bessel(x[i])
    jn_mx, jn_mx_prime = jn_bessel(m*x[i]) 
    yn_x, yn_x_prime = yn_bessel(x[i])
    yn_mx, yn_mx_prime = yn_bessel(m*x[i])

    a_n = (((m**2) * jn_mx * (x[i] * jn_prime_x)) - (jn_x * (m * x[i] * jn_mx_prime))) / (((m**2) * jn_mx * (x[i] * yn_x_prime)) - (yn_x * (m* x[i] * jn_mx_prime)))

    b_n = ((jn_mx * (x[i] * jn_prime_x)) - (jn_x * (m * x[i] * jn_mx_prime))) / ((jn_mx * (x[i] * yn_x_prime)) - (yn_x * (m* x[i] * jn_mx_prime)))

    a_n_array[i] = a_n
    b_n_array[i] = b_n

framewidth = 1.5
plt.rcParams['axes.linewidth'] = framewidth
plt.figure()
plt.plot(a_n_array, label='a_n', color='red')
plt.plot(b_n_array, label='b_n', color='blue')
plt.legend()
plt.xlabel(r'$x = 2 \pi r/ \lambda  \ [-]$',fontsize=12)
plt.ylabel(r'$a_n, b_n$',fontsize=12)
plt.title('Scattering coefficients', fontsize=12)
plt.xticks(fontsize=12), plt.yticks(fontsize=12)
plt.grid('True', linestyle='--', linewidth=0.5)

legend = plt.legend()
frame = legend.get_frame()
frame.set_linewidth(framewidth)
frame.set_edgecolor('black')

#%%

C_ext = (2 * np.pi / wavelength**2) * (n_array * np.abs(a_n_array + b_n_array))
#efficiency function 

Q_ext = C_ext / (np.pi * radius**2)


plt.figure()
# y log scale
plt.plot(radius, Q_ext, label='Q_ext', color='black')
plt.yscale('log')
plt.xlabel(r'$r \ [m]$',fontsize=12)
#%%

k = 200000 # cm-1
l = 2 * np.pi / k # cm
l = l * 1e7 # nm


print( l, 'nm')

print(k, 'nm-1')

#%% THAT IS THE FINAL CODE FOR THE SCATTERING COEFFICIENTS
radii = np.linspace (10,10000,1000)
wavenumber = 3.17687410e4  # cm-1
wavelength = (1e7) / wavenumber # nm

wavenumber = (1) / wavelength # nm-1

m = (1.366812 + 3.3249885e-9j) / 1

x = (2 * np.pi * radii ) / wavelength

area = np.pi * (radii**2)

C_ext = np.zeros(len(x))

sigmaSummation = np.zeros(len(x))

for i in range(len(x)):
    n_max = x[i] + 4 * x[i]**(1/3) 
    n_range = np.arange(1, n_max , 1).astype(int)

    for j in range(len(n_range)):
        jn_x = spherical_jn(n_range[j], x[i])
        jn_prime_x = spherical_jn(n_range[j], x[i], derivative=True)
        jn_mx = spherical_jn(n_range[j], m*x[i])
        jn_mx_prime = spherical_jn(n_range[j], m*x[i], derivative=True)
        yn_x = spherical_yn(n_range[j], x[i])
        yn_x_prime = spherical_yn(n_range[j], x[i], derivative=True)
        yn_mx = spherical_yn(n_range[j], m*x[i])
        yn_mx_prime = spherical_yn(n_range[j], m*x[i], derivative=True)

        hn_x = jn_x + 1j * yn_x
        hn_prime_x = jn_prime_x + 1j * yn_x_prime
        hn_mx = jn_mx + 1j * yn_mx
        hn_mx_prime = jn_mx_prime + 1j * yn_mx_prime
        

        a_n = (((m**2) * jn_mx * (jn_x + x[i] * jn_prime_x)) - (jn_x * ((m * jn_mx) + (m**2 * x[i]) * jn_mx_prime))) / (((m**2) * jn_mx * (hn_x + x[i] * hn_prime_x)) - (hn_x * ((m * jn_mx) + (m**2 * x[i]) * jn_mx_prime)))

        b_n = ((jn_mx * (jn_x + x[i] * jn_prime_x)) - (jn_x * ((m * jn_mx) + (m**2 * x[i]) * jn_mx_prime))) / ((jn_mx * (hn_x + x[i] * hn_prime_x)) - (hn_x * ((m * jn_mx) + (m**2 * x[i]) * jn_mx_prime)))

        # sigmaSummation[i] = sigmaSummation[i] + (2 * n_range[j] + 1) * np.sqrt((np.abs(a_n)**2 + np.abs(b_n)**2))
        sigmaSummation[i] = sigmaSummation[i] + (2 * np.pi / (wavenumber**2)) * (2 * n_range[j] + 1) * np.real(a_n + b_n)
    


# C = (2 * np.pi / (wavenumber**2)) * sigmaSummation
C_ext = sigmaSummation

Q_ext = C_ext / (area * (2*np.pi)**2)
#%%
framewidth = 1.5
plt.rcParams['axes.linewidth'] = framewidth
plt.figure()
plt.title(f'${radii[0]} \leq r \leq {radii[-1]} \ nm \  \lambda = {np.round(wavelength,2)} nm$', fontsize=12)
plt.plot(x, Q_ext, color='black',lw=1.5)
# plt.legend()
plt.xlabel(r'$x = 2 \pi r/ \lambda  \ [-]$',fontsize=12)
plt.ylabel(r'$Q_{ext} \ [-]$',fontsize=12)
plt.xticks(fontsize=12), plt.yticks(fontsize=12)
plt.grid('True', linestyle='--', linewidth=0.5)
# plt.xlim(-1.2, 60)
# legend = plt.legend()
# frame = legend.get_frame()
# frame.set_linewidth(framewidth)
# frame.set_edgecolor('black')
#%%
import numpy as np 
import matplotlib.pyplot as plt
# generate a gaussian distribution of integers with a mean of 500 and initial value of 100 and final value of 1000
mean = 500
std = 100
n = 1000
noOfP = np.random.normal(mean, std, n).astype(int)

# count the number of particles in each bin
bins = np.arange(100, 1000, 10)
hist, bins = np.histogram(noOfP, bins)

# plot the histogram
plt.figure()
plt.hist(noOfP, bins, edgecolor='black')
#%%
import numpy as np
import matplotlib.pyplot as plt 
# change the color of the plot

# Define parameters
D_min = 10   # Minimum particle diameter (nm)
D_max = 1000   # Maximum particle diameter (nm)
n_D = 100  # Number of particle size points
D = np.linspace(D_min, D_max, n_D)  # Particle size range (nm)
lambda_range = np.linspace(200, 1000, 100)  # Wavelength range (nm)
n_lambda = len(lambda_range)  # Number of wavelength point

# Discretize the particle size range
D_edges = np.linspace(D_min, D_max, n_D + 1)  # Bin edges
D_centers = 0.5 * (D_edges[:-1] + D_edges[1:])  # Bin centers

# Assume you have a particle size distribution vector V (same size as D_centers) : right skewed gaussian distribution
def particle_size_distribution(D):
    mean = 500  # Mean diameter (nm)
    std = 100  # Standard deviation (nm)
    V = np.exp(-0.5 * ((D - mean) / std)**2)
    return V 
def skewed_particle_size_distribution(D, mean, median, mode, skewness):
    from scipy.stats import skewnorm
    """
    Generate a particle size distribution skewed to the right or left.
    
    Parameters:
    - D: array-like, particle size values.
    - mean: desired mean of the distribution.
    - median: desired median of the distribution.
    - mode: desired mode of the distribution.
    - skewness: controls the direction and magnitude of skewness.
       * Positive skewness -> right-skewed.
       * Negative skewness -> left-skewed.
    
    Returns:
    - Array of distribution values evaluated at D.
    """
    # Fit skew-normal distribution parameters
    alpha = skewness  # Controls skewness
    loc = mode        # Approximate the mode (center point of the peak)
    scale = abs(mean - median)  # Approximate spread based on mean and median
    
    # Generate skewed distribution using scipy.stats.skewnorm
    V = skewnorm.pdf(D, a=alpha, loc=loc, scale=scale)
    V /= np.max(V)  # Normalize to 1
    return V

# Set distribution parameters
mean = 200 # Mean diameter (m)
median = 300 # It represents the 50th percentile
mode = 400 # it represents the peak of the distribution
skewness = -2 # Positive for right-skewed, negative for left-skewed
V = skewed_particle_size_distribution(D, mean, median, mode, skewness)

plt.figure()
plt.title('Particle size distribution', fontsize=12)
plt.plot(D, V, color='black',lw=2.5)
# add the mean, median and mode
# made the line to stop at the intersection of the curve
# Find the index of the closest value
mean_idx = np.argmin(np.abs(D - mean))
median_idx = np.argmin(np.abs(D - median))
mode_idx = np.argmin(np.abs(D - mode))

# Get the corresponding y-values
yMean = V[mean_idx]
yMedian = V[median_idx]
yMode = V[mode_idx]

# Plot the mean, median, and mode as a vertical line with a marker intersection with the curve
plt.plot([mean, mean], [0, yMean], color='red', linestyle='--', linewidth=1.5, zorder=1)
plt.plot([median, median], [0, yMedian], color='blue', linestyle='--', linewidth=1.5, zorder=1)
plt.plot([mode, mode], [0, yMode], color='green', linestyle='--', linewidth=1.5, zorder=1)

# Add text annotation
plt.text(mean, yMean, 'Mean', fontsize=12, color='red', ha='right', va='bottom')
plt.text(median, yMedian, 'Median', fontsize=12, color='blue', ha='right', va='bottom')
plt.text(mode, yMode, 'Mode', fontsize=12, color='green', ha='right', va='bottom')


plt.xlabel('Particle size $[nm]$', fontsize=12)
plt.ylabel('Normalized distribution $[-]$', fontsize=12)
framewidth = 1.5
plt.rcParams['axes.linewidth'] = framewidth
plt.xticks(fontsize=12), plt.yticks(fontsize=12)
plt.grid('True', linestyle='--', linewidth=0.5)
# legend = plt.legend(fontsize=12)    
# frame = legend.get_frame()
# frame.set_linewidth(framewidth)
# frame.set_edgecolor('black')



#%%
# Initialize the S matrix
S = np.zeros((n_lambda, n_D))

# Fill the S matrix
for i, lambda_ in enumerate(lambda_range):
    for j in range(n_D):
        # Integrate over the bin [D_j-1, D_j] using midpoint rule
        D_start, D_end = D_edges[j], D_edges[j+1]
        D_mid = D_centers[j]  # Midpoint
        Q = Q_ext(m=1.5, lambda_=lambda_, D=D_mid)  # Replace `m=1.5` with your refractive index
        S[i, j] = Q / D_mid * (D_end - D_start)

# Compute T = S @ V
T = S @ V

# Output results
print("S matrix shape:", S.shape)
print("Transmittance T shape:", T.shape)



#%% 
wl1 = 200   # nm
wl2 = 1000  # nm

wn1= 1e7 / wl1  # cm-1
wn2 = 1e7 / wl2 # cm-1

print (wn1, wn2) 

#%%
import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
cwd = os.getcwd()
# list all files in the directory
files = os.listdir(cwd)
# filter only the .csv files
files = [file for file in files if file.endswith('.csv')]

# Load the file: first row is the header and the delimiter is a comma
# do not read the commented lines
data = pd.read_csv("ri_water.csv", delimiter=",", comment='%', names=["Wave Number", "Real Part (n)", "Imaginary Part (k)"])
# Access columns
wave_number = data["Wave Number"] # cm-1
wavelength = 1e7 / wave_number # nm
real_part = data["Real Part (n)"]
imaginary_part = data["Imaginary Part (k)"]
m = real_part + 1j * imaginary_part
print(data.head())
#%%
framewidth = 1.5
plt.rcParams['axes.linewidth'] = framewidth
# Plot the refractive index of water
plt.figure()
plt.title('Complex refractive index of water: $m=n(\lambda) + ik(\lambda)$', fontsize=12)
plt.plot(wavelength , real_part, lw = framewidth,c='k')
plt.xlabel('Wavelength $[nm]$', fontsize=12)
plt.ylabel('Real part $n(\lambda)$', fontsize=12)    
plt.grid('True', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12), plt.yticks(fontsize=12)

plt.figure()
plt.title('Complex refractive index of water: $m=n(\lambda) + ik(\lambda)$', fontsize=12)
plt.plot(wavelength , imaginary_part, lw = framewidth,c='k')
plt.yscale('log')
plt.xlabel('Wavelength $[nm]$', fontsize=12)
plt.ylabel('Imaginary part $k(\lambda)$', fontsize=12)
plt.grid('True', which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12), plt.yticks(fontsize=12)
# legend = plt.legend(fontsize=12)    
# frame = legend.get_frame()
# frame.set_linewidth(framewidth)
# frame.set_edgecolor('black')
#%%
S = np.zeros((n_lambda, n_D))

radii = np.linspace (10,10000,1000)
wavenumber = 3.17687410e4  # cm-1
wavelength = (1e7) / wavenumber # nm

wavenumber = (1) / wavelength # nm-1

m = (1.366812 + 3.3249885e-9j) / 1

x = (2 * np.pi * radii ) / wavelength

area = np.pi * (radii**2)

C_ext = np.zeros(len(x))

sigmaSummation = np.zeros(len(x))

for i in range(len(x)):
    n_max = x[i] + 4 * x[i]**(1/3) 
    n_range = np.arange(1, n_max , 1).astype(int)

    for j in range(len(n_range)):
        jn_x = spherical_jn(n_range[j], x[i])
        jn_prime_x = spherical_jn(n_range[j], x[i], derivative=True)
        jn_mx = spherical_jn(n_range[j], m*x[i])
        jn_mx_prime = spherical_jn(n_range[j], m*x[i], derivative=True)
        yn_x = spherical_yn(n_range[j], x[i])
        yn_x_prime = spherical_yn(n_range[j], x[i], derivative=True)
        yn_mx = spherical_yn(n_range[j], m*x[i])
        yn_mx_prime = spherical_yn(n_range[j], m*x[i], derivative=True)

        hn_x = jn_x + 1j * yn_x
        hn_prime_x = jn_prime_x + 1j * yn_x_prime
        hn_mx = jn_mx + 1j * yn_mx
        hn_mx_prime = jn_mx_prime + 1j * yn_mx_prime
        

        a_n = (((m**2) * jn_mx * (jn_x + x[i] * jn_prime_x)) - (jn_x * ((m * jn_mx) + (m**2 * x[i]) * jn_mx_prime))) / (((m**2) * jn_mx * (hn_x + x[i] * hn_prime_x)) - (hn_x * ((m * jn_mx) + (m**2 * x[i]) * jn_mx_prime)))

        b_n = ((jn_mx * (jn_x + x[i] * jn_prime_x)) - (jn_x * ((m * jn_mx) + (m**2 * x[i]) * jn_mx_prime))) / ((jn_mx * (hn_x + x[i] * hn_prime_x)) - (hn_x * ((m * jn_mx) + (m**2 * x[i]) * jn_mx_prime)))

        # sigmaSummation[i] = sigmaSummation[i] + (2 * n_range[j] + 1) * np.sqrt((np.abs(a_n)**2 + np.abs(b_n)**2))
        sigmaSummation[i] = sigmaSummation[i] + (2 * np.pi / (wavenumber**2)) * (2 * n_range[j] + 1) * np.real(a_n + b_n)
    

# C = (2 * np.pi / (wavenumber**2)) * sigmaSummation
C_ext = sigmaSummation

Q_ext = C_ext / (area * (2*np.pi)**2)
#%%
import numpy as np
from scipy.special import spherical_jn, spherical_yn

def mie_C_ext(radius, wavelength, m):
    x = (2 * np.pi * radius ) / wavelength
    n_max = (x + 4 * x**(1/3)).astype(int)
    n_range = np.arange(1, n_max , 1).astype(int)
    C_ext = np.zeros_like(x)
    for j in range(len(n_range)):
        jn_x = spherical_jn(n_range[j], x)
        jn_prime_x = spherical_jn(n_range[j], x, derivative=True)
        jn_mx = spherical_jn(n_range[j], m*x)
        jn_mx_prime = spherical_jn(n_range[j], m*x, derivative=True)
        yn_x = spherical_yn(n_range[j], x)
        yn_x_prime = spherical_yn(n_range[j], x, derivative=True)
        yn_mx = spherical_yn(n_range[j], m*x)
        yn_mx_prime = spherical_yn(n_range[j], m*x, derivative=True)

        hn_x = jn_x + 1j * yn_x
        hn_prime_x = jn_prime_x + 1j * yn_x_prime
        hn_mx = jn_mx + 1j * yn_mx
        hn_mx_prime = jn_mx_prime + 1j * yn_mx_prime
        

        a_n = (((m**2) * jn_mx * (jn_x + x * jn_prime_x)) - (jn_x * ((m * jn_mx) + (m**2 * x) * jn_mx_prime))) / (((m**2) * jn_mx * (hn_x + x * hn_prime_x)) - (hn_x * ((m * jn_mx) + (m**2 * x) * jn_mx_prime)))

        b_n = ((jn_mx * (jn_x + x * jn_prime_x)) - (jn_x * ((m * jn_mx) + (m**2 * x) * jn_mx_prime))) / ((jn_mx * (hn_x + x * hn_prime_x)) - (hn_x * ((m * jn_mx) + (m**2 * x) * jn_mx_prime)))

        C_ext = C_ext + (2 * np.pi / (wavelength**2)) * (2 * n_range[j] + 1) * np.real(a_n + b_n)

    return C_ext

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.special import spherical_jn, spherical_yn

# radius = 10 
# wavelength = 500
# real_part = 1.33
# imaginary_part = 0.1
m = (1.366812 + 3.3249885e-9j) / 1
# m = real_part + 1j * imaginary_part
radii = np.linspace (10,10000,100)
wavelength = np.linspace(200,1000,200)

c_ext = np.zeros((200,100))
for i in range(200):
    for j in range(100):
        c_ext[i,j] = mie_C_ext(radii[j], wavelength[i], m)

#%%
Area = np.pi * (radii**2)
q_ext = c_ext / (Area * (2*np.pi)**2)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import spherical_jn, spherical_yn

list_of_files = os.listdir()
filename = 'ri_water.csv'
data = pd.read_csv(filename, delimiter=',', comment='%', names=["Wave Number", "Real Part (n)", "Imaginary Part (k)"])
wave_number = data["Wave Number"]
wavelength = 1e7 / wave_number
real_part = data["Real Part (n)"]
imaginary_part = data["Imaginary Part (k)"]

plt.figure()
plt.plot(wavelength, real_part, label='Real part', color='black')

#%%
walength_list = np.arange(200,1000,0.25)

# interpolate the real and imaginary part of the refractive index at the given wavelength

from scipy.interpolate import interp1d

f_real = interp1d(wavelength, real_part, kind='cubic')  
f_imag = interp1d(wavelength, imaginary_part, kind='cubic')

real_part_interpolated = f_real(walength_list)
imaginary_part_interpolated = f_imag(walength_list)

