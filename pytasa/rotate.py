
import numpy as num

pi = num.pi

# Copyright (c) 2011, James Wookey and Andrew Walker
# All rights reserved.
# 
# Redistribution and use in source and binary forms, 
# with or without modification, are permitted provided 
# that the following conditions are met:
# 
#    * Redistributions of source code must retain the 
#      above copyright notice, this list of conditions 
#      and the following disclaimer.
#    * Redistributions in binary form must reproduce 
#      the above copyright notice, this list of conditions 
#      and the following disclaimer in the documentation 
#      and/or other materials provided with the distribution.
#    * Neither the name of the University of Bristol nor the names 
#      of its contributors may be used to endorse or promote 
#      products derived from this software without specific 
#      prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS 
# AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL 
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def MS_rot3(C, alp, bet, gam, orderV=[0, 1, 2]):
    """
    MS_ROT3 - Elasticity matrix rotation.

     // Part of MSAT - The Matlab Seismic Anisotropy Toolkit //

     Rotates an elasticity matrix around the three axes. 

     Usage: 
         [CR] = MS_rot3(C,alp,bet,gam)                    
             C: input 6x6 elasticity matrix 
             alp: clockwise rotation about 1-axis, looking at origin
             bet: clockwise rotation about 2-axis, looking at origin
             gam: clockwise rotation about 3-axis, looking at origin

         [CR] = MS_rot3(...,'order',V)                    
             order: V is a 3 element vector containing the order in which to
                    apply the 3 rotations.
                    Default is [1 2 3].

     Notes:
         Angles are given in degrees and correspond to yaw, -dip and azimuth,
         respectvly. The rotations are applied in order, ie: alpha, then beta
         then gamma (by default). The variables C, alp, bet and gam can be 
         arrays or scalars but (1) the angles must all be the
         same length and (2) either C or all the angles must be scalars unless
         they are the same length.
    """

    # How many angles - are lists correct length.
    alp = num.array([alp])
    bet = num.array([bet])
    gam = num.array([gam])
    
    numrs = alp.shape[0]
    assert numrs == bet.shape[0]
    assert numrs == gam.shape[0]

    a = alp * pi/180.
    b = bet * pi/180.
    g = gam * pi/180.

    numdimCs = len(C.shape)
    if (numrs == 1) and (numdimCs == 2):
        # Two matrix scalar case
        CR = rot_3_scalar(C, a[0], b[0], g[0], orderV);  
    elif (numrs > 1) and (numdimCs == 2):
        # Many rotations and one matrix case
        CR = num.zeros((6,6,numrs));
        for i in range(numrs):
            CR[:,:,i] = rot_3_scalar(C, a[i], b[i], g[i], orderV)
    elif (numrs == 1) and (numdimCs == 3):
        # Many matrix and one rotation case
        CR = num.zeros((6, 6, C.shape[2]))
        for i in range(C.shape[2]):
            CR[:,:,i] = rot_3_scalar(C[:,:,i], a[0], b[0], g[0], orderV)
    elif (numrs > 1) and (numdimCs == 3):
        # List of rotation and matrices case
        assert numrs == C.shape[2]
        CR = num.zeros((6, 6, C.shape[2]))
        for i in range(C.shape[2]):
            CR[:,:,i] = rot_3_scalar(C[:,:,i], a[i], b[i], g[i], orderV)
    return CR


def rot_3_scalar(C, a, b, g, orderV):

    R = num.zeros((3, 3, 3))

    ca = num.cos(a)
    sa = num.sin(a)
    cb = num.cos(b)
    sb = num.sin(b)
    cg = num.cos(g)
    sg = num.sin(g)

    R = num.array([[[1, 0, 0],    [0, ca, sa],  [0, -sa, ca]],
                   [[cb, 0, -sb], [0, 1, 0],    [sb, 0, cb]],
                   [[cg, sg, 0],  [-sg, cg, 0], [0, 0, 1]]])

    RR = (R[orderV[2],:,:] @
          R[orderV[1],:,:] @
          R[orderV[0],:,:])
 
    # Delegate the rotation
    CR = MS_rotR(C, RR)

    return CR


def MS_rotR(C, R):
    """
    MS_ROTR - Script to rotate a set of elastic constants by a rotation matrix

    // Part of MSAT - The Matlab Seismic Anisotropy Toolkit //

    Rotate an elasticity matrices using a rotation matrix

     % [ CR ] = MS_rotR( C, R )

    Usage: 
        For a three-by-three rotation matrix, R, rotate the elasticity matrix,
        C to give a rotated matrix CR. 

    Notes:
       The rotation is performed without transforming the elasticity 
       matrix to the full tensor form following the method described in
       Bowers. This eleminates eight nested loops and replaces them with pure
       matrix-matrix operations, which is (~30 times) faster in Matlab. 
       Unlike the other MSAT rotation functions, C and R cannot be lists but
       must be 6x6 and 3x3 matricies, respectivly. Furthermore, the
       corectness of the input arguments are not checked. Users are encoraged
       to make use of MS_rot3 or MS_rotEuler for most rotation operations -
       these make use of this function internally.

    References:
       Bowers 'Applied Mechanics of Solids', Chapter 3

    See also: MS_ROT3 MS_ROTEULER
    """

    # form the K matrix
    # (based on Bowers 'Applied Mechanics of Solids', Chapter 3)
    K1 = num.array([[R[0,0]**2, R[0,1]**2, R[0,2]**2],
                    [R[1,0]**2, R[1,1]**2, R[1,2]**2],
                    [R[2,0]**2, R[2,1]**2, R[2,2]**2]])

    K2 = num.array([[R[0,1]*R[0,2], R[0,2]*R[0,0], R[0,0]*R[0,1]],
                    [R[1,1]*R[1,2], R[1,2]*R[1,0], R[1,0]*R[1,1]],
                    [R[2,1]*R[2,2], R[2,2]*R[2,0], R[2,0]*R[2,1]]])

    K3 = num.array([[R[1,0]*R[2,0], R[1,1]*R[2,1], R[1,2]*R[2,2]],
                    [R[2,0]*R[0,0], R[2,1]*R[0,1], R[2,2]*R[0,2]],
                    [R[0,0]*R[1,0], R[0,1]*R[1,1], R[0,2]*R[1,2]]])

    K4 = num.array([[R[1,1]*R[2,2] + R[1,2]*R[2,1], 
                     R[1,2]*R[2,0] + R[1,0]*R[2,2],
                     R[1,0]*R[2,1] + R[1,1]*R[2,0]],
                    [R[2,1]*R[0,2] + R[2,2]*R[0,1],
                     R[2,2]*R[0,0] + R[2,0]*R[0,2],
                     R[2,0]*R[0,1] + R[2,1]*R[0,0]],
                    [R[0,1]*R[1,2] + R[0,2]*R[1,1],
                     R[0,2]*R[1,0] + R[0,0]*R[1,2],
                     R[0,0]*R[1,1] + R[0,1]*R[1,0]]])

    K12 = num.concatenate((K1, 2*K2))
    K34 = num.concatenate((K3,   K4))
    K = num.concatenate((K12,   K34), axis=1)

    CR = K @ C @ K.T

    return CR
