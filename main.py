import arrayfire as af
import numpy as np
import os
from matplotlib import pyplot as pl

from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import wave_equation_2d
from dg_maxwell import wave_equation
from dg_maxwell import utils
from dg_maxwell import isoparam
from dg_maxwell import msh_parser

poly1_coeffs= af.reorder(\
            af.transpose(\
            af.np_to_af_array(np.array([[1, 2, 3., 4], [5, -2, -4.7211, 2]]))), 0, 2, 1)

poly2_coeffs = af.reorder(\
            af.transpose(\
            af.np_to_af_array(np.array([[-2, 4, 7., 9], [1, 0, -9.1124, 7]]))), 0, 2, 1)

test_array = (utils.polynomial_product_coeffs(poly1_coeffs, poly2_coeffs))[:, :, 0]
print(test_array)
print(utils.polyval_2d(test_array, params.xi_LGL, params.xi_LGL))
xi_i  = af.flat(af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL)))
eta_j = af.tile(params.xi_LGL, params.N_LGL)
f_ij  = np.e ** (xi_i + eta_j)
interpolated_f = wave_equation_2d.lag_interpolation_2d(f_ij)
xi  = utils.linspace(-1, 1, 8)
eta = utils.linspace(-1, 1, 8)
print(af.mean(af.abs(utils.polyval_2d(interpolated_f, xi, eta) - np.e**(xi+eta))))
