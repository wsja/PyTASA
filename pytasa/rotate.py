
import numpy as np

"""
Rotation operations for PyTASA

Copyright (c) 2011, James Wookey and Andrew Walker
All rights reserved.

Redistribution and use in source and binary forms, 
with or without modification, are permitted provided 
that the following conditions are met:

   * Redistributions of source code must retain the 
     above copyright notice, this list of conditions 
     and the following disclaimer.
   * Redistributions in binary form must reproduce 
     the above copyright notice, this list of conditions 
     and the following disclaimer in the documentation 
     and/or other materials provided with the distribution.
   * Neither the name of the University of Bristol nor the names 
     of its contributors may be used to endorse or promote 
     products derived from this software without specific 
     prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS 
AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL 
THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def rotate_C(C, alpha, beta, gamma, order=[0, 1, 2]):
    """
    Rotate elastic tensor or elasticity matrix aroun angles alpha, beta, gamma,
    in order.

    alpha: angle around x axis
    beta: angle around y axis
    gamma: angle around z axis
    order: (list of int) order to apply rotations
           [alpha, beta, gamma]
    """

    R = rotation_matrix(alpha, beta, gamma, order)

    if len(C.shape) == 2 and C.shape[0] == 6:
        #  Voigt notation
        C = rotate_Cij(C, R)
    elif len(C.shape == 4) and C.shape[0] == 3:
        # Tensor notation
        C = rotate_Cijkl(C, R)
    else:
        msg = "C must either be in Voigt or Tensor notation"
        raise TypeError(msg)

    return C


def rotate_Cij(C, R):
    """ Rotate elasticity matrix in Voigt notation """

    CT = cij2cijkl(C)
    CRT = rotate_Cijkl(CT, R)
    CR = cijkl2cij(CRT)
    return CR


def rotate_Cijkl(C, R):
    """Rotate tensor of forth rank around rotation matrix R"""
    RR = np.outer(R, R)
    RRRR = np.outer(RR, RR).reshape(4 * R.shape)
    axes = ((0, 2, 4, 6), (0, 1, 2, 3))
    CR = np.tensordot(RRRR, C, axes)
    return CR


def rotation_matrix(alpha, beta, gamma, order=[0, 1, 2]):
    """
    Create a rotation matrix from angles
    alpha: angle around x axis
    beta: angle around y axis
    gamma: angle around z axis
    order: (list of int) order to apply rotations
           [alpha, beta, gamma]
    """

    R = np.zeros((3, 3, 3))

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)

    R = np.array([[[1, 0, 0],    [0, ca, sa],  [0, -sa, ca]],
                  [[cb, 0, -sb], [0, 1, 0],    [sb, 0, cb]],
                  [[cg, sg, 0],  [-sg, cg, 0], [0, 0, 1]]])

    RR = np.matmul(R[order[2], :, :],
                   np.matmul(R[order[1], :, :],
                             R[order[0], :, :]))
    return RR


def ij2I(i, j):
    """
    Convert matrix indices to Voigt notation
    """
    if i == 0 and j == 0:
        return 0
    if i == 1 and j == 1:
        return 1
    if i == 2 and j == 2:
        return 2
    if (i == 1 and j == 2) or (i == 2 and j == 1):
        return 3
    if (i == 0 and j == 2) or (i == 2 and j == 0):
        return 4
    if (i == 0 and j == 1) or (i == 1 and j == 0):
        return 5


def cijkl2cij(C):
    """
    Convert from tensor Voigt elasticity matrix to tensor

    Converts between 3x3x3x3 tensor and a 6x6 Voigt representation of
    anisotropic elasticity. 

    Usage: 
        C must be rank 4 with size (3,3,3,3). The returned array C will be a
        rank 2 array with size (6,6).

    Notes:
        Do not use this function for the elastic compliance as additional
        terms are needed in this case.

    See also: cijkl2cij
    """
    CC = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    I = ij2I(i, j)
                    J = ij2I(k, l)
                    CC[I, J] = C[i, j, k, l]
    return CC


def cij2cijkl(C):
    """
    Convert from Voigt elasticity matrix to tensor

    Converts between a a 6x6 Voigt representation and a 3x3x3x3 tensor 
        representation of anisotropic elasticity. 

    Usage: 
        C must be a rank 2 array with size (6,6). The returned array C will 
        be rank 4 with size (3,3,3,3). 

    Notes:
        Do not use this function for the elastic compliance as additional
        terms are needed in this case.

    See also: cijkl2cij

    2005/07/04 - fixed Vera Schulte-Pelkum's bug
    """

    CC = np.zeros((3, 3, 3, 3))
    CC[0,0,0,0] = C[0,0]
    CC[1,1,1,1] = C[1,1]
    CC[2,2,2,2] = C[2,2]
    CC[1,2,1,2] = C[3,3]
    CC[2,1,2,1] = CC[1,2,1,2]
    CC[1,2,2,1] = CC[1,2,1,2]
    CC[2,1,1,2] = CC[1,2,1,2]
    CC[0,2,0,2] = C[4,4]    
    CC[2,0,0,2] = CC[0,2,0,2]
    CC[0,2,2,0] = CC[0,2,0,2]
    CC[2,0,2,0] = CC[0,2,0,2]
    CC[0,0,1,1] = C[0,1]    
    CC[1,1,0,0] = CC[0,0,1,1]
    CC[0,0,2,2] = C[0,2]    
    CC[2,2,0,0] = CC[0,0,2,2]
    CC[0,0,1,2] = C[0,3]    
    CC[0,0,2,1] = CC[0,0,1,2]
    CC[1,2,0,0] = CC[0,0,1,2]
    CC[2,1,0,0] = CC[0,0,1,2]
    CC[0,0,0,2] = C[0,4]    
    CC[0,0,2,0] = CC[0,0,0,2]
    CC[0,2,0,0] = CC[0,0,0,2]
    CC[2,0,0,0] = CC[0,0,0,2]
    CC[0,0,0,1] = C[0,5]    
    CC[0,0,1,0] = CC[0,0,0,1]
    CC[0,1,0,0] = CC[0,0,0,1]
    CC[1,0,0,0] = CC[0,0,0,1]
    CC[1,1,2,2] = C[1,2]    
    CC[2,2,1,1] = CC[1,1,2,2]
    CC[1,1,1,2] = C[1,3]    
    CC[1,1,2,1] = CC[1,1,1,2]
    CC[1,2,1,1] = CC[1,1,1,2]
    CC[2,1,1,1] = CC[1,1,1,2]
    CC[1,1,0,2] = C[1,4]    
    CC[1,1,2,0] = CC[1,1,0,2]
    CC[0,2,1,1] = CC[1,1,0,2]
    CC[2,0,1,1] = CC[1,1,0,2]
    CC[1,1,0,1] = C[1,5]    
    CC[1,1,1,0] = CC[1,1,0,1]
    CC[0,1,1,1] = CC[1,1,0,1]
    CC[1,0,1,1] = CC[1,1,0,1]
    CC[2,2,1,2] = C[2,3]     
    CC[2,2,2,1] = CC[2,2,1,2]
    CC[1,2,2,2] = CC[2,2,1,2]
    CC[2,1,2,2] = CC[2,2,1,2]
    CC[2,2,0,2] = C[2,4]
    CC[2,2,2,0] = CC[2,2,0,2]
    CC[0,2,2,2] = CC[2,2,0,2]
    CC[2,0,2,2] = CC[2,2,0,2]
    CC[2,2,0,1] = C[2,5]     
    CC[2,2,1,0] = CC[2,2,0,1]
    CC[0,1,2,2] = CC[2,2,0,1]
    CC[1,0,2,2] = CC[2,2,0,1]
    CC[1,2,0,2] = C[3,4]
    CC[2,1,0,2] = CC[1,2,0,2]
    CC[0,2,2,1] = CC[1,2,0,2]
    CC[0,2,1,2] = CC[1,2,0,2]
    CC[1,2,2,0] = CC[1,2,0,2]
    CC[2,1,2,0] = CC[1,2,0,2]
    CC[2,0,1,2] = CC[1,2,0,2]
    CC[2,0,2,1] = CC[1,2,0,2]
    CC[1,2,0,1] = C[3,5]
    CC[2,1,0,1] = CC[1,2,0,1]
    CC[0,1,1,2] = CC[1,2,0,1]
    CC[0,1,2,1] = CC[1,2,0,1]
    CC[1,2,1,0] = CC[1,2,0,1]
    CC[2,1,1,0] = CC[1,2,0,1]
    CC[1,0,1,2] = CC[1,2,0,1]
    CC[1,0,2,1] = CC[1,2,0,1]
    CC[0,2,0,1] = C[4,5]
    CC[2,0,0,1] = CC[0,2,0,1]
    CC[0,1,0,2] = CC[0,2,0,1]
    CC[0,1,2,0] = CC[0,2,0,1]
    CC[0,2,1,0] = CC[0,2,0,1]
    CC[2,0,1,0] = CC[0,2,0,1]
    CC[1,0,0,2] = CC[0,2,0,1]
    CC[1,0,2,0] = CC[0,2,0,1]
    CC[0,1,0,1] = C[5,5]
    CC[1,0,0,1] = CC[0,1,0,1]
    CC[0,1,1,0] = CC[0,1,0,1]
    CC[1,0,1,0] = CC[0,1,0,1]
    return CC
