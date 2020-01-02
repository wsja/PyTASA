#! /usr/bin/python3

import numpy as num

"""
Compute long wavelength equivalent anisotropic elastic medium
"""

eps = abs(7./3 - 4./3 - 1)

def backus_monoclin(f=[], c=[]):
    """
    Compute long wavelength anisotorpic medium from anisotropic layers
    f: (list(float)) Volume fractions (must add up to 1)
    c: (list(Cij)) elasticity tensors (unit of pressure)
    
    Source:
    Kumar (2013) J. Geophys. Eng.
    Applying Backus averaging for deriving seismic anisotropy of a
    long-wavelength equivalent medium from well-log data
    """


    if len(c) != len(f):
        msg = 'c, f, and m must have the same size and have at least one element'
        raise IndexError(msg)

    f = num.array(f)
    if abs(num.sum(f) - 1) > eps:
        msg = 'Elements of f must sum up to 1'
        raise ValueError(msg)

    av = lambda x: num.sum(f*x)  # average
    get = lambda i, j: num.array([cc[i-1, j-1] for cc in c])

    c11 = get(1, 1)
    c22 = get(2, 2)
    c33 = get(3, 3)
    c44 = get(4, 4)
    c55 = get(5, 5)
    c66 = get(6, 6)

    c12 = get(1, 2)
    c13 = get(1, 3)
    c15 = get(1, 5)
    c23 = get(2, 3)
    c25 = get(2, 5)
    c35 = get(3, 5)
    c46 = get(4, 6)

    A = c33*c55 - c35**2
    AA = (av(c33/A)*av(c55/A) - av(c35/A)**2)**-1
    B1 = (c13*c55 - c15*c35)/A
    B2 = (c15*c33 - c13*c35)/A
    B3 = (c23*c55 - c25*c35)/A
    B4 = (c25*c33 - c23*c35)/A

    B5 = AA*(av(c33/A)*av(B1)**2 + 2*av(c35/A)*av(B1)*av(B2) +
             av(c55/A)*av(B2)**2)

    B6 = AA*(av(c33/A)*av(B1)*av(B3) + av(c35/A)*av(B1)*av(B4) +
             av(c35/A)*av(B2)*av(B3) + av(c55/A)*av(B2)*av(B4))

    B7 = AA*(av(c33/A)*av(B3)**2 + 2*av(c35/A)*av(B3)*av(B4) +
             av(c55/A)*av(B4)**2)

    ce = num.zeros((6, 6))
    ce[0, 0] = av(c11) - av(c13*B1 + c15*B2) + B5
    ce[1, 1] = av(c22) - av(c23*B3 + c25*B4) + B7
    ce[2, 2] = AA*av(c33/A)
    ce[3, 3] = av(1/c44)**-1
    ce[4, 4] = AA*av(c55/A)
    ce[5, 5] = av(c66) - av(c46**2/c44) + av(1/c44)**-1 * av(c46/c44)**2

    ce[0, 1] = av(c12) - av(c13*B3 + c15*B4) + B6
    ce[0, 2] = AA*av(c33/A)*av(B1) + AA*av(c35/A)*av(B2)
    ce[0, 4] = AA*av(c35/A)*av(B1) + AA*av(c55/A)*av(B2)
    ce[1, 0] = ce[0, 1]
    ce[2, 0] = ce[0, 2]
    ce[4, 0] = ce[0, 4]

    ce[1, 2] = AA*av(c33/A)*av(B3) + AA*av(c35/A)*av(B4)
    ce[1, 4] = AA*av(c35/A)*av(B3) + AA*av(c55/A)*av(B4)
    ce[2, 1] = ce[1, 2]
    ce[4, 1] = ce[1, 4]


    ce[2, 4] = AA*av(c35/A)
    ce[4, 2] = ce[2, 4]

    ce[3, 5] = av(1/c44)**-1 * av(c46/c44)
    ce[5, 3] = ce[3, 5]

    return ce
