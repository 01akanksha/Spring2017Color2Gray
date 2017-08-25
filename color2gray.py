import os
import sys
import argparse
from numpy import linalg as LA
import numpy as np
import scipy
from scipy import io, misc, sparse
from skimage import color

PIC_DIR = 'pics'
EXTENSION = '.png'

alpha = 8
theta = np.pi/4
v_theta = np.asarray((np.cos(theta), np.sin(theta)))

def dot_prod(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        dot_prod(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def crunch(x):
    return alpha * np.tanh(x / alpha)

def neighbour_pixels(pos_mat, r, mu=1):
    d = mu
    l1, h1 = max(r[0]-d, 0), min(r[0]+d, pos_mat.shape[0])
    l2, h2 = max(r[1]-d, 0), min(r[1]+d, pos_mat.shape[1])
    return pos_mat[l1:h1 + 1, l2:h2 + 1]

def delta_calculation(col_lab_result, r, S):
    values = []
    for s in S:
        delta_luminance = col_lab_result[r][0] - col_lab_result[s][0]
        delta_chrominance = col_lab_result[r][1:] - col_lab_result[s][1:]
        delta_cnorm_res = LA.norm(delta_chrominance)
        cval = crunch(delta_cnorm_res)
        dotprod = np.dot(v_theta, delta_chrominance)
        if abs(delta_luminance) > cval:
            values.append(delta_luminance)
        elif dotprod >= 0:
            values.append(cval)
        else:
            values.append(crunch(-delta_cnorm_res))
    return values

def do_optimization(delta, size):
    A = np.ones((size, size), dtype='float32') * -1.0
    for i in range(size):
        A[i, i] = size
    B = np.sum(delta, axis=0).transpose() - np.sum(delta, axis=1)
    return A, B

def create_delta(col_lab_result, list_neighbour, arrxy2idx, height, width):
    size = height * width
    delta =  sparse.lil_matrix((size, size))
    for i in range(height):
        for j in range(width):
            cur_index = arrxy2idx[i, j]
            neighbors = list_neighbour[cur_index]
            n_index = [ arrxy2idx[pos] for pos in neighbors ]
            cur_val = delta_calculation(col_lab_result, (i, j), neighbors)
            delta[cur_index, n_index] = np.asmatrix(cur_val)
    return delta.todense()	
	
def main(argts):
    
    par = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    par.add_argument('--input', help='Specify the input picture.', type=str, default='sunrise')

    args = par.parse_args(argts)
    col_name = args.input

    input_path = os.path.join(PIC_DIR, col_name + EXTENSION)
    output_path = os.path.join(PIC_DIR, col_name + '_bw' + EXTENSION)

    col_rgb = misc.imread(input_path)
    col_lab_result = color.rgb2lab(col_rgb[:, :, 0:3])

    (height, width) = col_rgb.shape[0:2]
    size = height * width
    
    gray_matrix = np.average(col_rgb, axis=2)
    
    cart = dot_prod([range(height), range(width)])
    cart_r = cart.reshape(height, width, 2) 
    arrxy2idx = np.arange(size).reshape(height, width)
    list_neighbour = [] 
    for i in range(height):
        for j in range(width):
            cur_index = arrxy2idx[i, j]
            cur_neighbor = neighbour_pixels(cart_r, [i, j], max(height, width)).reshape(-1, 2)
            cur_neighbor = [ tuple(item) for item in cur_neighbor ]
            cur_neighbor.remove((i, j))
            list_neighbour.append(cur_neighbor)
    delta_matrix = create_delta(col_lab_result, list_neighbour, arrxy2idx, height, width)
    
    A, B = do_optimization(delta_matrix, size)
    best_g = np.linalg.solve(A, -B)
    g_matrix = best_g.reshape(height, width)
    gray_matrix = np.zeros((height, width, 3))
    for i in range(3):
        gray_matrix[:, :, i] = g_matrix

    misc.imsave(output_path, gray_matrix)
    print('Color picture saved to', output_path)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))