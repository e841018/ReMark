import pickle
import numpy as np, cv2, tqdm
import ml

################################################################################
# conventions
################################################################################

# axis 1 (x axis) is rightward
# axis 2 (y axis) is downward
# angles are counterclockwise

################################################################################
# detection stage
################################################################################

def draw_detection_spec(res, max_n_blob=4, min_dist=2):
    h, w = res
    min_dist_pw2 = min_dist ** 2
    n_blob = np.random.randint(1, max_n_blob + 1)
    specs = []
    while len(specs) < n_blob:
        y = np.random.uniform(0, h)
        x = np.random.uniform(0, w)
        too_close = False
        for y_, x_, i_ in specs:
            if (y - y_) ** 2 + (x - x_) ** 2 < min_dist_pw2:
                too_close = True
                break
        if too_close:
            continue
        intensity = np.random.uniform(0, 1)
        specs.append((y, x, intensity))
    return specs

def synth_detection_img(res, specs, sigma=0.8):
    ''' synthesize a marker in the detection stage

    ### parameters:
    *   `res`: (h, w) in macropixels
    *   `specs`: list of (y, x, intensity), y, x in macropixels, intensity in [0, 1]
    *   `sigma`: float, sigma of Gaussian function

    ### returns:
    *   `img`: dtype=np.float32, shape=res, values in [0, 1]
    '''
    h, w = res
    img = np.zeros(res, dtype=np.float32)
    for y, x, intensity in specs:
        Y = np.arange(h).reshape(h, 1)
        X = np.arange(w).reshape(1, w)
        img += intensity * np.exp((-0.5 / sigma ** 2) * ((X-x) ** 2 + (Y-y) ** 2))
    return img

if __name__ == '__main__':

    n_draw = 100000
    res = 18, 32

    all_data = []
    while len(all_data) < n_draw:
        specs = draw_detection_spec(res)
        img = synth_detection_img(res, specs)
        if img.sum() > 0.001:
            all_data.append((img, ))

    dataset_name = f'clear_detection_sample={n_draw}'
    dataset_dir = 'dataset/recon_detection/'
    with open(f'../{dataset_dir}/{dataset_name}.pkl', 'wb') as f:
        pickle.dump(all_data, f)
    print(f'created dataset {dataset_name}.pkl\n    in {dataset_dir}')

################################################################################
# identification stage
################################################################################

def draw_content(n_draw, s=4):
    contents = []
    for i in range(n_draw):
        contents.append(np.random.randint(0, 2, (s, s)))
    return contents

def calc_aff_mat(x, y, deg, edge_len, tilt_axis, tilt_deg):
    ''' calculate an affine transformation matrix from geometrical parameters

    ### returns:
    *   `aff_mat`: as defined in embed.cnm_embed(), unified, dtype=np.float32, shape=(2, 3)
    '''
    angle1 = (deg - tilt_axis) / 180 * np.pi
    angle2 = tilt_axis / 180 * np.pi
    c1, s1 = np.cos(angle1), np.sin(angle1)
    c2, s2 = np.cos(angle2), np.sin(angle2)
    c1c2 = c1 * c2
    c1s2 = c1 * s2
    s1c2 = s1 * c2
    s1s2 = s1 * s2
    scale_keep = edge_len * 0.5
    scale_shrink = scale_keep * np.cos(tilt_deg / 180 * np.pi)
    # bases = [[ c2, s2],  @  [[scale_keep,            0],  @  [[ c1, s1],
    #          [-s2, c2]]      [         0, scale_shrink]]      [-s1, c1]]
    bases = np.array([[ c1c2,  s1c2], [-c1s2, -s1s2]]) * scale_keep \
          + np.array([[-s1s2,  c1s2], [-s1c2,  c1c2]]) * scale_shrink
    aff_mat = np.zeros((2, 3), dtype=np.float32)
    aff_mat[:, :2] = bases
    aff_mat[0, 2] = x
    aff_mat[1, 2] = y
    aff_mat = ml.embed.cnm_unify(aff_mat)
    return aff_mat

def check_boundary(aff_mat, res):
    ''' check if any of the vertices is out of boundary after transform

    ### parameters:
    *   `aff_mat`: as defined in embed.cnm_embed()
    *   `res`: (h, w)

    ### returns:
    *   True if all vertices are in boundary
    '''
    h, w = res
    # marker at center
    vertices = np.array([[-1, -1,  1,  1],
                         [-1,  1, -1,  1],
                         [ 1,  1,  1,  1]])
    # affine transformation
    vertices_tf = aff_mat @ vertices
    if np.any(vertices_tf[0] < 0):
        return False
    if np.any(vertices_tf[0] > w-1):
        return False
    if np.any(vertices_tf[1] < 0):
        return False
    if np.any(vertices_tf[1] > h-1):
        return False
    return True

def draw_aff_mat(n_draw, res, max_tilt=60):
    h, w = res
    min_res = min(res)
    edge_len_min = min_res * 0.4
    edge_len_max = min_res * 0.6 # There is a typo in Table 1 of the paper. `19.6` should be corrected to `19.2`.

    aff_mats = []
    n_chunk = 128
    while len(aff_mats) < n_draw:
        x = np.random.normal(loc=(w-1)/2, scale=0.5*(w-min_res*0.65), size=n_chunk) # position offset
        y = np.random.normal(loc=(h-1)/2, scale=0.5*(h-min_res*0.65), size=n_chunk) # position offset
        deg = np.random.uniform(low=0., high=90., size=n_chunk) # rotation angle
        edge_len = np.random.uniform(low=edge_len_min, high=edge_len_max, size=n_chunk) # edge length
        tilt_axis = np.random.uniform(low=0., high=180., size=n_chunk) # tilt axis orientation
        tilt_deg = np.random.uniform(low=0., high=max_tilt, size=n_chunk) # tilt angle
        for param in zip(x, y, deg, edge_len, tilt_axis, tilt_deg):
            aff_mat = calc_aff_mat(*param)
            if check_boundary(aff_mat, res):
                aff_mats.append(aff_mat)
    return aff_mats[:n_draw]

def draw_bias_spec(n_draw, max_radius=1100):
    slope = 0.055 / 60 # change in relative intensity per pixel
    size = 10, 10 # macropixel size
    shape = 32, 32 # number of macropixels
    shift_y = size[0] * shape[0] / 2
    shift_x = size[1] * shape[1] / 2

    bias_specs = []
    n_chunk = 128
    while len(bias_specs) < n_draw:
        y = np.random.uniform(low=-max_radius, high=max_radius, size=n_chunk).astype(np.float32)
        x = np.random.uniform(low=-max_radius, high=max_radius, size=n_chunk).astype(np.float32)
        radius = np.sqrt(y**2 + x**2)
        deg = -np.arctan2(y, x) * (180 / np.pi)
        # notation of corners a, b, c, d:
        # +-------+--------+-------+
        # |   a   |   top  |   b   |
        # +-------+--------+-------+
        # |  left | (y, x) | right |
        # +-------+--------+-------+
        # |   c   | bottom |   d   |
        # +-------+--------+-------+
        t = y - shift_y
        b = y + shift_y
        l = x - shift_x
        r = x + shift_x
        radius_a = np.sqrt(t**2 + l**2)
        radius_b = np.sqrt(t**2 + r**2)
        radius_c = np.sqrt(b**2 + l**2)
        radius_d = np.sqrt(b**2 + r**2)
        val_min = 1 - np.max([radius_a, radius_b, radius_c, radius_d], axis=0) * slope
        val_max = 1 - np.min([radius_a, radius_b, radius_c, radius_d], axis=0) * slope
        for i in range(n_chunk):
            if radius[i] < max_radius:
                bias_specs.append((deg[i], (val_min[i], val_max[i])))
    return bias_specs[:n_draw]

def render_bias(res, deg, val_range=(0.0, 1.0)):
    ''' render a bias parameter

    ### parameters:
    *   `res`: (h, w)
    *   `deg`: direction of gradient vectors. 0 deg is right, 90 deg is up.
    *   `val_range`: (val_min, val_max), where val_min could be negative

    ### returns:
    *   `bias`: dtype=np.float32, shape=res, values in val_range, nonnegative
    '''
    # grid
    h, w = res
    y_grid = np.arange(0, h)
    x_grid = np.arange(0, w)
    X, Y = np.meshgrid(x_grid, y_grid)
    # rotation
    rad = deg / 180 * np.pi
    c = np.cos(rad)
    s = np.sin(rad)
    bias = (c*X - s*Y).astype(np.float32)
    # scale
    bias_min, bias_max = np.min(bias), np.max(bias)
    val_min, val_max = val_range
    scale = (val_max-val_min) / (bias_max-bias_min)
    bias = bias * scale + (- scale * bias_min + val_min)
    bias[bias < 0] = 0
    return bias

def synth_identification_img(res, content, aff_mat, sigma=0.8, bias=None):
    ''' synthesize a marker in the identification stage

    ### parameters:
    *   `res`: (h, w)
    *   `content`: shape=(4, 4), value in {0, 1}
    *   `aff_mat`: as defined in embed.cnm_embed(), dtype=np.float32, shape=(2, 3)
    *   `sigma`: float, sigma of cv2.GaussianBlur
    *   `bias`: dtype=np.float32, shape=res, values in [0, 1]

    ### returns:
    *   `img`: dtype=np.float32, shape=res, values in [0, 1]
    '''
    # generate marker at center
    L = 12 # marker size
    marker = np.ones((L, L), dtype=np.uint8)
    marker[1:-1, 1:-1] = 0
    marker[2:-2, 2:-2] = np.kron(content, np.ones((2, 2), dtype=content.dtype))
    marker *= 255
    # normalize coordinates to [-1, 1]
    l = (L - 1) / 2
    M = np.array([[ 1/l,   0,  -1],
                  [   0, 1/l,  -1],
                  [   0,   0,   1]], dtype=np.float32)
    # affine transform
    M = aff_mat @ M
    img = cv2.warpAffine(marker, M, res[::-1], flags=cv2.INTER_CUBIC)
    # Gaussian blur
    if sigma != 0:
        img = cv2.GaussianBlur(img, (5, 5), sigma)
    # normalize values to the range [0, 1]
    img = (img / 255).astype(np.float32)
    # bias
    if bias is not None:
        img *= bias
    return img

if __name__ == '__main__':

    n_draw = 100000
    res = 32, 32

    n_draw_spare = int(n_draw * 1.01)
    contents = draw_content(n_draw_spare)
    aff_mats = draw_aff_mat(n_draw_spare, res)
    bias_specs = draw_bias_spec(n_draw_spare)
    all_data = []
    for i in tqdm.tqdm(range(n_draw_spare)):
        img = synth_identification_img(
            res,
            contents[i],
            aff_mats[i],
            bias=render_bias(res, *bias_specs[i]))
        if img.sum() > 0.001:
            all_data.append((img, aff_mats[i], bias_specs[i]))
    assert len(all_data) >= n_draw, len(all_data)
    all_data = all_data[:n_draw]

    dataset_name = f'clear_identification_sample={n_draw}'
    dataset_dir = 'dataset/recon_identification/'
    with open(f'../{dataset_dir}/{dataset_name}.pkl', 'wb') as f:
        pickle.dump(all_data, f)
    print(f'created dataset {dataset_name}.pkl\n    in {dataset_dir}')
