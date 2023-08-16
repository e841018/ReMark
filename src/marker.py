import numpy as np, cv2, matplotlib, pickle

################################################################################
# extract cells from image
################################################################################

def extract_cells(recon, aff_mat, alpha=0.15, return_extracted_Z=False):
    '''
    extract cell contents
    parameters:
        recon: shape=(32, 32), dtype=np.float32
        aff_mat: as defined in cnm_embed()
        alpha: float >= 0, coeffecient for deconvolution
        return_extracted_Z: return the cells before polarization and estimated
            bias
    return:
        cells: shape=(4, 4), dtype=np.float32
        extracted: shape=(4, 4), dtype=np.float32
        Z: shape=(4, 4), dtype=np.float32
    '''
    # dimensions
    gran = 4 # granularity, number of times to divide a pixel
    L = 12 # side length of a patch
    H, W = 32, 32

    # upsample recon
    recon = np.kron(recon, np.ones((gran, gran), dtype=recon.dtype))

    # generate a patch for each cell and the rim
    empty_patch = np.zeros((L, L), dtype=np.uint8)
    patches = []
    for i in range(4):
        for j in range(4):
            patch = empty_patch.copy()
            patch[2+i*2: 4+i*2, 2+j*2: 4+j*2] = 255
            patches.append(patch)
    patch = empty_patch.copy() + 255
    patch[1:-1, 1:-1] = 0
    patches.append(patch)
    
    # affine transform the patches
    l = (L - 1) / 2
    M = aff_mat @ np.array([[1/l,   0, -1],
                            [  0, 1/l, -1],
                            [  0,    0, 1]])
    # shift by [0.5, 0.5], enlarge by gran, and shift by [-0.5, -0.5]
    if gran != 1:
        M *= gran
        M [:, 2] += (gran - 1) / 2
    dsize = (W * gran, H * gran)
    binary_imgs = []
    for patch in patches:
        patch = cv2.warpAffine(patch, M, dsize, flags=cv2.INTER_NEAREST)
        patch = cv2.GaussianBlur(patch, (5, 5), 0.8)
        patch = patch > 230
        binary_imgs.append(patch)
    masks, rim = binary_imgs[:-1], binary_imgs[-1]

    # fit the rim to a 2D plane: aX+bY+c
    # [ x0 y0 1 ]         [z0]
    # [ x1 y1 1 ]   [a]   [z1]
    # [ :  :  : ] @ [b] = [: ]
    # [ :  :  : ]   [c]   [: ]
    # [ :  :  : ]         [: ]
    X, Y = np.meshgrid(range(W * gran), range(H * gran))
    X_sub = X[rim][:,np.newaxis]
    Y_sub = Y[rim][:,np.newaxis]
    Z_sub = recon[rim][:,np.newaxis]
    mat = np.hstack((X_sub, Y_sub, np.ones((len(X_sub), 1)))).astype(np.float32)
    a, b, c = (np.linalg.pinv(mat) @ Z_sub)[:, 0]
    Z = (a * X + b * Y + c).astype(np.float32)
    Z[Z < 0] = 0

    # downsample
    recon = np.array([np.sum(recon[mask]) for mask in masks]).reshape(4, 4)
    Z = np.array([np.sum(Z[mask]) for mask in masks]).reshape(4, 4)

    # deconvolve recon
    accu = np.zeros((4, 4), dtype=recon.dtype)
    accu[:, :3] += recon[:, 1:]
    accu[:, 1:] += recon[:, :3]
    accu[:3, :] += recon[1:, :]
    accu[1:, :] += recon[:3, :]
    recon -= alpha * accu
    recon[recon < 0] = 0

    # polarize recon: bright cells still white, dark cells become negative
    # polarize(recon) = recon - abs(recon - Z), where Z is the estimated bias
    # the reason is because the following two are equivalent:
    # 1. minimize_{dict_cells} sum(abs(recon_cells - Z*dict_cells))
    # 2. maximize_{dict_cells} sum(polarize(recon_cells) * dict_cells)
    # The second one is more efficient, especially when dictionary size is large
    cells = recon - np.abs(recon-Z) # dist_to_dark - dist_to_bright

    if return_extracted_Z:
        return cells, recon, Z
    else:
        return cells

################################################################################
# annotate parallelogram
################################################################################

tf_rotate = matplotlib.transforms.Affine2D().rotate_deg(-135)

def poly_parallelogram(aff_mat, t, b, l, r, color, tf=None, **kwargs):
    '''
    parameters:
        aff_mat: see `add_parallelograms()`
        t, b, l, r in [-1, 1], relative to the center lines of white rims
        color: matplotlib.patches.Polygon(..., edgecolor=color)
        tf: None or matplotlib.transforms.Affine2D
        kwargs: passed to matplotlib.patches.Polygon()
    return:
        matplotlib.patches.Polygon
    '''
    square = np.array([[t, t, b, b],
                       [l, r, r, l],
                       [1, 1, 1, 1]])
    parallelogram = (aff_mat @ square).T

    if tf is not None:
        parallelogram = tf.transform(parallelogram)
    return matplotlib.patches.Polygon(
        parallelogram, closed=True, facecolor='none', edgecolor=color, **kwargs)

def add_parallelograms(ax, aff_mat, tf=False, no_upper_left=False, **kwargs):
    '''
    add parallelograms in ax
    parameters:
        ax: pyplot axis
        aff_mat: as defined in cnm_embed()
        tf: if True transforms with tf_rotate
        kwargs: passed to matplotlib.patches.Polygon()
    '''
    tf = tf_rotate if tf else None
    ax.add_patch(poly_parallelogram(aff_mat, -1, 1, -1, 1, 'red', tf, **kwargs))
    sub_space = 1/11
    sub_size = 4/11 - sub_space
    sub_shift = 4/11
    for i in [-1.5, -0.5, 0.5, 1.5]:
        t, b = np.array([-0.5, 0.5]) * sub_size + i*sub_shift
        for j in [-1.5, -0.5, 0.5, 1.5]:
            l, r = np.array([-0.5, 0.5]) * sub_size + j*sub_shift
            ax.add_patch(poly_parallelogram(
                aff_mat, t, b, l, r, 'cyan', tf, **kwargs))
    # annotate the upper left square
    if not no_upper_left:
        ax.add_patch(poly_parallelogram(
            aff_mat, -8/11, -4/11, -8/11, -4/11, 'yellow', tf, **kwargs))

################################################################################
# identify
################################################################################

with open('../data/markers_aruco_DICT_4X4_1000.pkl', 'rb') as f:
    markers_aruco = np.array(pickle.load(f)).reshape(-1, 16)

def augment(img):
    return img, \
        np.rot90(img), \
        np.rot90(img, k=2), \
        np.rot90(img, k=3)

def identify_polarized(observed_cells, dict_size=1000):
    corr_max = -float('inf') # default value
    for cells in augment(observed_cells):
        cells = cells.reshape(-1)
        corrs = np.dot(markers_aruco[:dict_size], cells)
        m = np.argmax(corrs)
        corr = corrs[m]
        if corr > corr_max:
            corr_max = corr
            m_max = m
    return m_max
