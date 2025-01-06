# list of functions for Mie scattering theory

import numpy as np
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
        yn_mx = spherical_yn(n_range[j], m*x)
        yn_mx_prime = spherical_yn(n_range[j], m*x, derivative=True)

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
def get_RI(original_RI_real, original_RI_imag, wavelength, kind = 'cubic'):
    # interpolate the real and imaginary parts of the refractive index
    RI_real_interp = interp1d(wavelength, original_RI_real, kind = kind)
    RI_imag_interp = interp1d(wavelength, original_RI_imag, kind = kind)

    return RI_real_interp, RI_imag_interp



