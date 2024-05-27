import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy import optimize
import numpy as np
import math
from mpi4py import MPI
import time
import pandas as pd
from mpi4py.util import pkl5
# Manipulate channels
def extract_rgb(img):
    return img[:,:,0], img[:,:,1], img[:,:,2]

def assemble_rbg(img_r, img_g, img_b):
    shape = (img_r.shape[0], img_r.shape[1], 1)
    return np.concatenate((np.reshape(img_r, shape), np.reshape(img_g, shape),
        np.reshape(img_b, shape)), axis=2)

# Transformations

def reduce(img, factor):
    result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j] = np.mean(img[i*factor:(i+1)*factor,j*factor:(j+1)*factor])
    return result

def rotate(img, angle):
    return ndimage.rotate(img, angle, reshape=False)

def flip(img, direction):
    return img[::direction,:]

def apply_transformation(img, direction, angle, contrast=1.0, brightness=0.0):
    return contrast*rotate(flip(img, direction), angle) + brightness

# Contrast and brightness
def find_contrast_and_brightness1(D, S):
    contrast = 0.75
    brightness = (np.sum(D - contrast*S)) / D.size
    return contrast, brightness

def find_contrast_and_brightness2(D, S):
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    return x[1], x[0]

def generate_all_transformed_blocks(img, source_size, destination_size, step):
    factor = source_size // destination_size
    transformed_blocks = []
    for k in range((img.shape[0] - source_size) // step + 1):
        for l in range((img.shape[1] - source_size) // step + 1):
            S = reduce(img[k*step:k*step+source_size,l*step:l*step+source_size], factor)
            # Generate all possible transformed blocks
            for direction, angle in candidates:
                transformed_blocks.append((k, l, direction, angle, apply_transformation(S, direction, angle)))
    return transformed_blocks

def compress(img, source_size, destination_size, step):
    transformations = []
    transformed_blocks = generate_all_transformed_blocks(img, source_size, destination_size, step)
    i_count = img.shape[0] // destination_size
    j_count = img.shape[1] // destination_size
    for i in range(i_count):
        transformations.append([])
        for j in range(j_count):
           # print("{}/{} ; {}/{}".format(i, i_count, j, j_count))
            transformations[i].append(None)
            min_d = float('inf')
            D = img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size]
            for k, l, direction, angle, S in transformed_blocks:
                contrast, brightness = find_contrast_and_brightness2(D, S)
                S = contrast*S + brightness
                d = np.sum(np.square(D - S))
                if d < min_d:
                    min_d = d
                    transformations[i][j] = (k, l, direction, angle, contrast, brightness)
    return transformations

def decompress(transformations, source_size, destination_size, step, nb_iter=8):
    factor = source_size // destination_size
    height = len(transformations) * destination_size
    width = len(transformations[0]) * destination_size
    iterations = [np.random.randint(0, 256, (height, width))]
    cur_img = np.zeros((height, width))
    for i_iter in range(nb_iter):
       # print(i_iter)
        for i in range(len(transformations)):
            for j in range(len(transformations[i])):
                # Apply transform
                k, l, flip, angle, contrast, brightness = transformations[i][j]
                S = reduce(iterations[-1][k*step:k*step+source_size,l*step:l*step+source_size], factor)
                D = apply_transformation(S, flip, angle, contrast, brightness)
                cur_img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size] = D
        iterations.append(cur_img)
        cur_img = np.zeros((height, width))
    return iterations

# Compression for color images
def reduce_rgb(img, factor):
    img_r, img_g, img_b = extract_rgb(img)
    img_r = reduce(img_r, factor)
    img_g = reduce(img_g, factor)
    img_b = reduce(img_b, factor)
    return assemble_rbg(img_r, img_g, img_b)

def compress_rgbMPI(img, source_size, destination_size, step, rank, size):
    img_r, img_g, img_b = extract_rgb(img)
    if rank == 0:
      compress1 = compress(img_r, source_size, destination_size, step)
      compress2 = compress(img_g, source_size, destination_size, step)
      return compress1, compress2
    elif rank == 1:
        compress2 = compress(img_g, source_size, destination_size, step)
        return compress2
    elif rank == 2:
        compress3 = compress(img_b, source_size, destination_size, step)
        return compress3

def compress_rgb(img, source_size, destination_size, step):
    img_r, img_g, img_b = extract_rgb(img)
    return [compress(img_r, source_size, destination_size, step),
        compress(img_g, source_size, destination_size, step),
        compress(img_b, source_size, destination_size, step)]

def decompress_rgb(transformations, source_size, destination_size, step, nb_iter=8):
    img_r = decompress(transformations[0], source_size, destination_size, step, nb_iter)[-1]
    img_g = decompress(transformations[1], source_size, destination_size, step, nb_iter)[-1]
    img_b = decompress(transformations[2], source_size, destination_size, step, nb_iter)[-1]
    return assemble_rbg(img_r, img_g, img_b)

directions = [1, -1]
angles = [0, 90, 180, 270]
candidates = [[direction, angle] for direction in directions for angle in angles]

# cd C:\Users\den19\PycharmProjects\pythonProject2
# mpiexec -n 3 python MPI.py

def test_rgb(rank, size):
    global transformations
    img = mpimg.imread("C:/Users/den19/Downloads/lena.gif")
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.array(img).astype(np.uint8), interpolation='none')
    img = reduce_rgb(img, 8)
    if rank == 0:
      transformations1 = compress_rgbMPI(img, 8, 4, 8, rank, size)
      ts1 = transformations1
      comm.irecv(transformations1, source=1, tag = 71)
      ts2 = transformations1
      comm.irecv(transformations1, source=2, tag = 72)
      ts3 = transformations1
      transformations = [ts1, ts2, ts3]
      retrieved_img = decompress_rgb(transformations1, 8, 4, 8)
      plt.figure()
      plt.subplot(122)
      plt.imshow(np.array(retrieved_img).astype(np.uint8), interpolation='none')
      plt.show()
    elif rank == 1:
      transformations2 = compress_rgbMPI(img, 8, 4, 8, rank, size)
      comm.isend(transformations2, dest=0, tag = 71)

    elif rank == 2:
      transformations3 = compress_rgbMPI(img, 8, 4, 8, rank, size)
      comm.isend(transformations3, dest=0, tag = 72)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #print("%d of %d" % (comm.Get_rank(), comm.Get_size()))
    if (rank == 0):
     start = time.time()

    test_rgb(comm.Get_rank(), comm.Get_size())

    if (rank == 0):
     end = time.time()
     print("The time of execution of above program is :", (end - start), "s")
