import numpy as np

################################################################################
# conventions
################################################################################

# axis 1 (x axis) is rightward
# axis 2 (y axis) is downward
# angles are counterclockwise

################################################################################
# CNM (continuous embedding)
################################################################################

cnm_scale = 32

def angle_double(x, y):
    X = x ** 2
    Y = y ** 2
    r = np.sqrt(X + Y)
    return (X - Y) / r, 2 * x * y / r

def angle_half(x, y):
    # [-180, 180) -> [-90, 90)
    if x < 0 and y == 0:
        return 0, -x
    r = np.sqrt(x ** 2 + y ** 2)
    return np.sqrt(r * (r+x) / 2), np.sqrt(r * (r-x) / 2) * np.sign(y)

def ab2nm(Ax, Ay, Bx, By):
    # transform 1
    A_D = np.array(angle_double(Ax, Ay))
    B_D = np.array(angle_double(Bx, By))
    # transform 2
    Mx, My = (A_D + B_D) / 2
    Dx, Dy = (A_D - B_D) / 2
    # transform 3
    Nx, Ny = angle_double(Dx, Dy)
    return Nx, Ny, Mx, My

def nm2ab(Nx, Ny, Mx, My):
    # transform 3
    M = np.array((Mx, My))
    D = np.array(angle_half(Nx, Ny))
    # transform 2
    A_D = M + D
    B_D = M - D
    # transform 1
    Ax, Ay = angle_half(*A_D)
    Bx, By = angle_half(*B_D)
    return Ax, Ay, Bx, By

def cnm_embed(aff_mat):
    ''' embed an affine transformation matrix aff_mat
        [[Ax, Bx, Cx],
         [Ay, By, Cy]]

    into [Cx, Cy, Nx, Ny, Mx, My]

    The transformation is from the marker coordinates:
    *   The origin is at center of the marker.
    *   Basis vectors are ending at the centers of the right and bottom edges (center of white rims, not outer borders).

    to the image coordinates:
    *   The origin is at the center of the upper-left macropixel.
    *   Basis vectors are rightward and downward, lengths are 1 macropixel.

    ### parameters:
    *   `aff_mat`: as defined above

    ### returns:
    *   `CNM`: [Cx, Cy, Nx, Ny, Mx, My], dtype=np.float32
    '''
    (Ax, Bx, Cx), (Ay, By, Cy) = aff_mat / cnm_scale
    Nx, Ny, Mx, My = ab2nm(Ax, Ay, Bx, By)
    return np.array([Cx, Cy, Nx, Ny, Mx, My], dtype=np.float32)

def cnm_extract(CNM):
    ''' extract aff_mat from [Cx, Cy, Nx, Ny, Mx, My]

    The extracted aff_mat is unified, i.e., guaranteed to satisfy:
    *   (1) angle(A), angle(B) in [-90, 90)
    *   (2) angle(A) - angle(B) in [0, 180]

    ### parameters:
    *   `CNM`: [Cx, Cy, Nx, Ny, Mx, My]

    ### returns:
    *   `aff_mat`: as defined in embed.cnm_embed(), unified, dtype=np.float32, shape=(2, 3)
    '''
    Cx, Cy, Nx, Ny, Mx, My = CNM
    Ax, Ay, Bx, By = nm2ab(Nx, Ny, Mx, My)
    AcrossB = Ax*By - Ay*Bx
    if AcrossB < 0:
        Ax, Ay, Bx, By = Bx, By, Ax, Ay
    aff_mat = np.array([
        [Ax, Bx, Cx],
        [Ay, By, Cy]
    ], dtype=np.float32)
    aff_mat *= cnm_scale
    return aff_mat

def cnm_unify(aff_mat):
    ''' equivalent to cnm_extract(cnm_embed(aff_mat))

    ### parameters:
    *   `aff_mat`: as defined in embed.cnm_embed(), shape=(2, 3)

    ### returns:
    *   `aff_mat`: as defined in embed.cnm_embed(), unified, dtype=`aff_mat`.dtype, shape=(2, 3)
    '''
    (Ax, Bx, Cx), (Ay, By, Cy) = aff_mat

    # angle(A) in [-90, 90)
    if Ax < 0:
        Ax, Ay = -Ax, -Ay
    if Ax == 0 and Ay < 0:
        Ay = -Ay

    # angle(B) in [-90, 90)
    if Bx < 0:
        Bx, By = -Bx, -By
    if Bx == 0 and By < 0:
        By = -By

    # angle(A) - angle(B) in [0, 180]
    AcrossB = Ax*By - Ay*Bx
    if AcrossB < 0:
        Ax, Ay, Bx, By = Bx, By, Ax, Ay

    # construct new aff_mat
    aff_mat = np.array([
        [Ax, Bx, Cx],
        [Ay, By, Cy]
    ], dtype=aff_mat.dtype)
    return aff_mat

################################################################################
# CAB (naive discontinuous embedding)
################################################################################

cab_scale = 32

def cab_embed(aff_mat):
    (Ax, Bx, Cx), (Ay, By, Cy) = aff_mat / cab_scale
    return np.array([Cx, Cy, Ax, Ay, Bx, By], dtype=np.float32)

def cab_extract(CAB):
    Cx, Cy, Ax, Ay, Bx, By = CAB
    aff_mat = np.array([
        [Ax, Bx, Cx],
        [Ay, By, Cy]
    ], dtype=np.float32)
    aff_mat *= cab_scale
    return aff_mat

################################################################################
# global switch
################################################################################

embed = cnm_embed
extract = cnm_extract
