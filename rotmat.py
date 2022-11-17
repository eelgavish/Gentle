import numpy as np
import math
## Axis Angle -> RMat
# Description: Returns the corresponding rotation matrix of an axis and
#              angle.
#
# Output: Rout | SO3 Rotation Matrix   | 3x3 array
#
# Input:     w | Unit Axis of Rotation | 3x1 array
#           th | Angle of Rotation     | radians
#
# Created by: Ethan Elgavish | 01-29-2021
# Edited by: Ethan Elgavish | 03-01-2022
# Ported to Python by: Ethan Elgavish | 11-04-2022
def AARotMat(w: np.array, th):
    Rout = np.eye(3)  # Error output
    # Check if w has the correct number of elements
    if np.shape(w) != (3,):
        print('-FATAL- AARotMat: Input w has the wrong number of elements, aborting.')
        return
    # Check if w is a unit vector
    if math.sqrt(sum(np.multiply(w,w))) != 1:
        #print('-ERROR- AARotMat: Input w is not a unit vector, normalizing and proceeding.') 
        w = np.divide(w, np.linalg.norm(w))
    # Catch statement for no rotation case
    if th == 0:
        #print('-INFO- AARotMat: Input theta is zero, no rotation.') 
        return
    elif th > math.pi:
        #print('-ERROR- AARotMat: Input theta is greater than pi, subtracting pi and proceeding.') 
        th = th # math.pi
    elif th < -math.pi:
        #print('-ERROR- AARotMat: Input theta is less than -pi, adding pi and proceeding.') 
        th = -(-th % math.pi) 
    
    # Pre-calculation for simplification
    C = math.cos(th) 
    S = math.sin(th) 
    t = 1 - C 
    # Calculation of rotation matrix
    Rout = np.array([[t*w[0]**2+C,t*w[0]*w[1]-S*w[2],t*w[0]*w[2]+S*w[1]],
      		         [t*w[0]*w[1]+S*w[2],t*w[1]**2+C,t*w[1]*w[2]-S*w[0]],
        		     [t*w[0]*w[2]-S*w[1],t*w[1]*w[2]+S*w[0],t*w[2]**2+C]])
    if checkRotation(Rout) == False:
        print("-ERROR- AARotMat: Output Rotation Matrix is NOT SO(3).")
    return Rout

## Rotation Matrix Checker
# Description: Check that input is an SO(3) rotation matrix
#
# Output: bool | Boolean value   | boolean
#
# Input:     R | Rotation Matrix | 3x3 SO(3) Matrix
#
# Created by: Ethan Elgavish | 02-24-2022
# Ported to Python by: Ethan Elgavish | 11-04-2022
def checkRotation(R):
    # Check that input is an SO(3) rotation matrix
    boo = False
    if np.round(R*R.T).astype(int).all() == np.eye(3).all() and int(np.round(np.linalg.det(R))) == 1:
        boo = True
    return boo

## Skew : Vector -> Skew Symmetric Matrix
# Description: Spans a 1x3 vector into a skew symmetric matrix
#
# Output: ssw | Skew Symetric Matrix   | 3x3 Skew Symmetric Matrix
#
# Input:    w | Unskewed Vector        | 3x1 array
#
# Created by: Ethan Elgavish | 02-15-2022
# Edited by: Ethan Elgavish | 03-01-2022
def skew(w):
    # Check if w has the correct number of elements
    if np.shape(w) != (3,):
        print('-FATAL- skew: Input w has the wrong number of elements, aborting.')
        return
    return np.array([[0,-w[2],w[1]], [w[2],0,-w[0]], [-w[1],w[0],0]])

## Align
# Description: Find Rotation Matrix to Align two Vectors in 3D space
#
# Output: R | Rotation Matrix | 3x3 SO(3) Matrix
#
# Input:  a | First Vector    | 3x1 Vector
#         b | Second Vector   | 3x1 Vector
#
# Created by: Ethan Elgavish | 02-24-2022
# Ported to Python by: Ethan Elgavish | 11-04-2022
def align(a,b):
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    R = np.eye(3)+skew(v)+np.dot(skew(v),skew(v))*(1-c)/(s**2)
    if checkRotation(R):
        return R
    else:
        print("Not SO(3)")
        return R