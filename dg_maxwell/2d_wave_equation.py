#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import isoparam

def dx_dxi(x_nodes, xi, eta):
    '''
    Computes the derivative :math:`\\frac{\\partial x}{\\partial \\xi}`.
    The derivative is obtained by finding the derivative of the analytical
    function :math:`x \\equiv x(\\xi, \\eta)`.
    
    Parameters
    ----------
    x_nodes : np.ndarray [8]
              :math:`x` nodes.
              
    xi      : float
            :math:`\\xi` coordinate for which :math:`x` has to be found.

    eta     : float
            :math:`\\eta` coordinate for which :math:`x` has to be found.
            
    Returns
    -------
    dx_dxi : float
             :math:`\\frac{\\partial x}{\\partial \\xi}` calculated at
             :math:`(\\xi, \\eta)` coordinate.
    '''
    
    dN_0_dxi = -0.25*eta**2 + (0.5*eta + 0.5)*xi - 0.25*eta
    dN_1_dxi = 0.5*eta**2 - 0.5
    dN_2_dxi = -0.25*eta**2 + (-0.5*eta + 0.5)*xi + 0.25*eta
    dN_3_dxi = (eta - 1.0)*xi
    dN_4_dxi = 0.25*eta**2 + (-0.5*eta + 0.5)*xi - 0.25*eta
    dN_5_dxi = -0.5*eta**2 + 0.5
    dN_6_dxi = 0.25*eta**2 + (0.5*eta + 0.5)*xi + 0.25*eta
    dN_7_dxi = (-1.0*eta - 1.0)*xi
    
    dx_dxi = dN_0_dxi * x_nodes[0] \
           + dN_1_dxi * x_nodes[1] \
           + dN_2_dxi * x_nodes[2] \
           + dN_3_dxi * x_nodes[3] \
           + dN_4_dxi * x_nodes[4] \
           + dN_5_dxi * x_nodes[5] \
           + dN_6_dxi * x_nodes[6] \
           + dN_7_dxi * x_nodes[7]

    return dx_dxi

def dx_deta(x_nodes, xi, eta):
    """
    This function will numerically compute the derivative
    dx_deta.
    """
    
    dN_0_deta = (-0.5*eta - 0.25)*xi + 0.25*xi**2 + 0.5*eta
    dN_1_deta = eta*xi - 1.0*eta
    dN_2_deta = (-0.5*eta + 0.25)*xi - 0.25*xi**2 + 0.5*eta
    dN_3_deta = 0.5*xi**2 - 0.5
    dN_4_deta = (0.5*eta - 0.25)*xi - 0.25*xi**2 + 0.5*eta
    dN_5_deta = -1.0*eta*xi - 1.0*eta
    dN_6_deta = (0.5*eta + 0.25)*xi + 0.25*xi**2 + 0.5*eta
    dN_7_deta = -0.5*xi**2 + 0.5

    dx_deta = dN_0_deta * x_nodes[0] \
            + dN_1_deta * x_nodes[1] \
            + dN_2_deta * x_nodes[2] \
            + dN_3_deta * x_nodes[3] \
            + dN_4_deta * x_nodes[4] \
            + dN_5_deta * x_nodes[5] \
            + dN_6_deta * x_nodes[6] \
            + dN_7_deta * x_nodes[7]

    return dx_deta

def dy_dxi(y_nodes, xi, eta):
    """
    This function will numerically compute the derivative
    dy_dxi.
    """
    return dx_dxi(y_nodes, xi, eta)

def dy_eta(y_nodes, xi, eta):
    """
    This function will numerically compute the derivative
    dy_deta.
    """
    return dx_deta(y_nodes, xi, eta)