# list of functions for Mie scattering theory
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn
from scipy.interpolate import interp1d
def mie_C_ext(radius, wavelength, m):
    x = (2 * np.pi * radius ) / wavelength # size parameter [-]
    n_max = (x + 4 * x**(1/3)).astype(int) # maximum number of terms in the series
    n_range = np.arange(1, n_max , 1).astype(int) # range of terms in the series
    C_ext = np.zeros_like(x) # initialize the extinction cross section
    for j in range(len(n_range)):
        jn_x = spherical_jn(n_range[j], x) # spherical Bessel function of the first kind
        jn_prime_x = spherical_jn(n_range[j], x, derivative=True) # derivative of the spherical Bessel function of the first kind
        jn_mx = spherical_jn(n_range[j], m*x) 
        jn_mx_prime = spherical_jn(n_range[j], m*x, derivative=True)
        yn_x = spherical_yn(n_range[j], x) # spherical Bessel function of the second kind
        yn_x_prime = spherical_yn(n_range[j], x, derivative=True) # derivative of the spherical Bessel function of the second kind

        hn_x = jn_x + 1j * yn_x # spherical Hankel function of the first kind
        hn_prime_x = jn_prime_x + 1j * yn_x_prime # derivative of the spherical Hankel function of the first kind
        
        # calculate the coefficients a_n and b_n
        a_n = (((m**2) * jn_mx * (jn_x + x * jn_prime_x)) - (jn_x * ((m * jn_mx) + (m**2 * x) * jn_mx_prime))) / (((m**2) * jn_mx * (hn_x + x * hn_prime_x)) - (hn_x * ((m * jn_mx) + (m**2 * x) * jn_mx_prime)))

        b_n = ((jn_mx * (jn_x + x * jn_prime_x)) - (jn_x * ((m * jn_mx) + (m**2 * x) * jn_mx_prime))) / ((jn_mx * (hn_x + x * hn_prime_x)) - (hn_x * ((m * jn_mx) + (m**2 * x) * jn_mx_prime)))


        # calculate the extinction cross section
        C_ext = C_ext + (2 * np.pi / (wavelength**2)) * (2 * n_range[j] + 1) * np.real(a_n + b_n)

    C_ext = C_ext / ((2*np.pi)**2) # unit is m^2
        
    return C_ext

# calculate the geometric cross section
def mie_C_geo(radius):
    C_geo = np.pi * radius**2 # geometric cross section
    return C_geo

# extract the real and imaginary parts of the refractive index at a given wavelength
def get_RI(original_RI_real, original_RI_imag, wavelength, kind, wavelengthOfInterest):
    # interpolate the real and imaginary parts of the refractive index
    RI_real_interp = interp1d(wavelength, original_RI_real, kind = kind, fill_value = 'extrapolate')
    RI_imag_interp = interp1d(wavelength, original_RI_imag, kind = kind, fill_value = 'extrapolate')

    n = RI_real_interp(wavelengthOfInterest) # real part of the refractive index at the wavelength of interest
    k = RI_imag_interp(wavelengthOfInterest) # imaginary part of the refractive index at the wavelength of interest
    return n, k

def particle_size_distribution(D, mean, std):
    V = np.exp(-0.5 * ((D - mean) / std)**2)
    return V 

#%%

particles_radii = np.arange(10,10000,step = 75)

wavelenghts = np.arange(200,1000,step = 4)

import pandas as pd
from tqdm import tqdm
# read the refractive index of water with pandas
ri_water = pd.read_csv('ri_water.csv', delimiter = ',',comment = '%')

wavelength_ri = ri_water.iloc[:,0].values
#make it in nm from cm-1
wavelength_ri = 1e7/wavelength_ri

n = ri_water.iloc[:,1].values
k = ri_water.iloc[:,2].values 


#%%
n_oi, k_oi = get_RI(n, k, wavelength_ri,'cubic', wavelenghts)
m = n_oi + 1j*k_oi

C_ext = np.zeros((len(wavelenghts), len(particles_radii)))
for i in tqdm(range(len(wavelenghts)), desc="Wavelengths"):
    for j in range(len(particles_radii)):
        C_ext[i,j] = mie_C_ext(particles_radii[j], wavelenghts[i], m[i])


#%%

mean = 4000
std = 1000
PSD = particle_size_distribution(particles_radii, mean, std)

plt.figure()
plt.plot(particles_radii, PSD)
plt.xlabel('Particle size (nm)')
plt.ylabel('Volume fraction')

#%%
Q_ext = C_ext/ mie_C_geo(particles_radii)

S = np.zeros((len(wavelenghts), len(particles_radii)))
# Calculate particle diameters from radii
D_centers = 2 * particles_radii  # Convert radii to diameters
n_D = len(D_centers)   # Number of particle size bins
n_lambda = Q_ext.shape[0]  # Number of wavelengths

# Assume you also have the edges of the particle size bins
D_edges = np.zeros(n_D + 1)  # Create edges array
D_edges[:-1] = D_centers - (D_centers[1] - D_centers[0]) / 2  # Approximate edges
D_edges[-1] = D_centers[-1] + (D_centers[1] - D_centers[0]) / 2
# Construct the S matrix
for i in range(len(wavelenghts)):  # Loop over wavelengths
    for j in range(len(particles_radii)):  # Loop over particle size bins
        # Directly use Q_ext_matrix and the bin widths
        D_start, D_end = D_edges[j], D_edges[j+1]
        S[i, j] = Q_ext[i, j] / D_centers[j] * (D_end - D_start)


#%%
# lhs = -(2/(3*L))* np.log(T)
# make PSD to ve a column vector
#  check if the matrix multiplication is possible 
print(S.shape)
print(PSD.shape)
# shape is (200,134) and (134,1) so we can multiply them. We will get a matrix of shape (200,1)

lhs = S @ PSD

# path length 
L = 0.095 # m 

# adjust the path length unit
L = L * 1e9 # nm
T = np.exp(-(3*L/2) * lhs)

# noise: random white noise

np.random.seed(0)
noise = np.random.normal(0, 2e-11, len(lhs))


Tnoisy = np.exp(-(3*L/2) * (lhs + noise))


plt.figure()
plt.plot(Tnoisy)

lhsNoisy = -(2/(3*L)) * np.log(Tnoisy)
plt.figure()
plt.plot(lhsNoisy)


#%%
TNoisy = np.exp(-(3*L/2) * lhsNoisy)

plt.figure()
plt.plot(wavelenghts, T)
plt.xlabel('Wavelength (nm)')

plt.plot(wavelenghts, TNoisy)

#%% 
# in this section we use the least square method to solve an inverse problem 
#  Ax = b 
# the test case is A= S, x = PSD, b = lhs
# we will use the numpy.linalg.lstsq method to solve the problem
#%%
# A = S
# b = lhs
# x = PSD
# solve the inverse problem
# PSD_est, residuals, rank, s = np.linalg.lstsq(S, lhs, rcond=None)

#  calculate the condition number of the matrix S
cond_num = np.linalg.cond(S)
print(f'The condition number of the matrix S is {cond_num}')

# calculate the rank of the matrix S
rank = np.linalg.matrix_rank(S)
print(f'The rank of the matrix S is {rank}')

#%%
# compute the svd of the matrix S
U, s, Vh = np.linalg.svd(S)

# plot the singular values
plt.figure()
plt.plot(s)
plt.yscale('log')
plt.xlabel(r'$n$')
plt.ylabel(r'$\sigma_n$')
plt.title('Singular values of the matrix S')

# add a log grid to the plot
plt.grid("minor", linestyle='--', linewidth=0.5)


# check the first n singular vector
plt.figure()
plt.subplot(3,3,1)
plt.title(r'$u_1$')
plt.plot(U[:,0])
plt.grid("both", linestyle='--', linewidth=0.5)

plt.subplot(3,3,2)
plt.title(r'$u_2$')
plt.plot(U[:,1])
plt.grid("both", linestyle='--', linewidth=0.5)

plt.subplot(3,3,3)
plt.title(r'$u_3$')
plt.plot(U[:,2])
plt.grid("both", linestyle='--', linewidth=0.5)

plt.subplot(3,3,4)
plt.title(r'$u_4$')
plt.plot(U[:,3])
plt.grid("both", linestyle='--', linewidth=0.5)

plt.subplot(3,3,5)
plt.title(r'$u_5$')
plt.plot(U[:,4])
plt.grid("both", linestyle='--', linewidth=0.5)

plt.subplot(3,3,6)
plt.title(r'$u_6$')
plt.plot(U[:,5])
plt.grid("both", linestyle='--', linewidth=0.5)

plt.subplot(3,3,7)
plt.title(r'$u_7$')
plt.plot(U[:,6])
plt.grid("both", linestyle='--', linewidth=0.5)

plt.subplot(3,3,8)
plt.title(r'$u_8$')
plt.plot(U[:,7])
plt.grid("both", linestyle='--', linewidth=0.5)

plt.subplot(3,3,9)
plt.title(r'$u_9$')
plt.plot(U[:,8])
plt.grid("both", linestyle='--', linewidth=0.5)


plt.tight_layout()

# Picard plot
# 1) transforma the lhs into SVD basis
# compute the Fourier coefficients
coeff = np.dot(U.T, lhs)

# prepare data for the Picard plot
sigma = s
rhs_coeff = np.abs(coeff[:len(sigma)])

#%%
plt.figure()
plt.plot(sigma)
plt.plot(rhs_coeff)
plt.yscale('log')
plt.grid("both", linestyle='--', linewidth=0.5)
plt.xlabel(r'$n$')
plt.ylabel(r'$\sigma_n$')
plt.title('Picard plot')

#%%
# 2-norm condition number of the matrix S
norm_2 = np.linalg.norm(S, 2)
norm_inv = np.linalg.norm(np.linalg.pinv(S), 2)
cond_num_2 = norm_2 * norm_inv
condSigma = s[0] / s[-1]

#%% Tikhnov regularization

# solve the inverse problem using the Tikhnov regularization
# the Tikhnov regularization is a method to solve the inverse problem by adding a regularization term to the least square problem

# specifically the Tikhnov solution is defined as the solution of the following problem
# x = argmin(||Ax - b||^2 + alpha * ||x||^2)
# where alpha is a regularization parameter

# we will use the numpy.linalg.lstsq method to solve the problem

# define the regularization parameter

alpha = 0.6e-21
x = np.linalg.solve(S.T @ S + alpha * np.eye(S.shape[1]), S.T @ lhsNoisy)

plt.figure()
plt.plot(particles_radii, x)
plt.plot(particles_radii, PSD)

#%% non-negative Philipps-Twomey regularization
# solve the inverse problem using the non-negative Philipps-Twomey regularization
# the essence of the Philipps-Twomey regularization is the suppression of small singular values by introducing a filter factor
#fi = s**2 / (s**2 + gamma**2)

# where gamma is a regularization parameter

# we will use the numpy.linalg.lstsq method to solve the problem

#next step is to constrain the solution to be non-negative as the PSD cannot be negative

# 

#%%
n = 134
# reconstruct the matrix S using the first n singular values
S_reconstructed = U[:, :n] @ np.diag(s[:n]) @ Vh[:n, :]

# solve the inverse problem using the reconstructed matrix S
PSD_est, residuals, rank, s = np.linalg.lstsq(S_reconstructed, lhs, rcond=None)

plt.figure()
plt.plot(particles_radii, PSD_est)
plt.xlabel('Particle size (nm)')
plt.ylabel('Volume fraction')

# from the previous analysis we can say that to recontruct the matrix S we need to use all the singular values. This 