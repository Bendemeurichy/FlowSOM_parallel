from __future__ import annotations

from os import cpu_count
from typing import Literal

import numpy as np
from numba import jit, prange
from numbasom import SOM as numbaSOM
from xpysom import XPySom


@jit(nopython=True, parallel=False)
def eucl(p1, p2):
    distance = 0.0
    for j in range(len(p1)):
        diff = p1[j] - p2[j]
        distance += diff * diff
    return np.sqrt(distance)


@jit(nopython=True, parallel=True)
def manh(p1, p2):
    return np.sum(np.abs(p1 - p2))


@jit(nopython=True)
def chebyshev(p1, p2, px, n, ncodes):
    distance = 0.0
    for j in range(px):
        diff = abs(p1[j * n] - p2[j * ncodes])
        if diff > distance:
            distance = diff
    return distance


@jit(nopython=True, parallel=True)
def cosine(p1, p2, px, n, ncodes):
    nom = 0.0
    denom1 = 0.0
    denom2 = 0.0
    for j in range(px):
        nom += p1[j * n] * p2[j * ncodes]
        denom1 += p1[j * n] * p1[j * n]
        denom2 += p2[j * ncodes] * p2[j * ncodes]

    return (-nom / (np.sqrt(denom1) * np.sqrt(denom2))) + 1


def SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf=eucl, seed=None,
        version: Literal['numba', 'xpysom', 'original', 'lr'] = 'numba', batch=False,
        batch_size=0):
    """SOM function used in som_estimator.py.

    Wrapper function to call either the numba or xpysom implementation of the SOM algorithm, used to train the SOM
    model.
    Can also call the original implementation of the SOM algorithm or the batch implementation of the SOM.
    :param data: The data to train the SOM model on
    :param codes: The initial codes for the SOM model
    :param nhbrdist: The neighbourhood distance
    :param alphas: The learning rates
    :param radii: The radii
    :param ncodes: The number of codes
    :param rlen: The number of iterations
    :param distf: The distance function to use
    :param seed: The random seed to use
    :param version: The version of the SOM algorithm to use
    :type version: Literal['numba', 'xpysom','original','lr'] Can be either 'numba', 'original',
        'lr'(original with different learning rate) or 'xpysom', defaults to 'numba'.
        Xpysom uses the batch implementation of the SOM algorithm.
    :param batch: If True, the batch version of the SOM algorithm will be used
    :param batch_size: The batch size to use
    :return: The trained SOM model.
    """
    if batch and not (version == 'xpysom' or version == 'my_batch'):
        raise ValueError('Batch version of the SOM algorithm is only available in the xpysom implementation')
    if version == 'numba':
        return calculate_numbaSOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf, seed)
    elif version == 'xpysom':
        if batch and batch_size == 0:
            batch_size = data.shape[0] // cpu_count()
        return calculate_xpySOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf, seed, batch_size)
    elif version == 'original':
        return calculate_originalSOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf, seed)
    elif version == 'lr':
        return lr_SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf, seed)
    elif version == 'my_batch':
        if batch_size == 0:
            batch_size = data.shape[0] // cpu_count()
        return calculate_SOM_batch(data, codes, nhbrdist, alphas, radii, ncodes, rlen, batch_size, distf, seed)

    raise ValueError('version should be either numba or xpysom')


def calculate_xpySOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf=eucl, seed=None, batch_size=0):
    """Calculate the SOM using the xpysom implementation.

    This code comes from the public GitHub repository https://github.com/Manciukic/xpysom by user Manciukic.
    The parameters are the same as the SOM function in this file.
    You could use a different distance function.
    Since
    they don't really work for the original SOM implementation,
    I have used the standard euclidian distance function.
    """
    if seed is not None:
        np.random.seed(seed)

    # check if batch is wanted
    if batch_size == 0:
        batch_size = data.shape[0]
    # find the dimensions of the data
    xdim = int(np.sqrt(ncodes))
    pysom = XPySom(xdim, xdim, data.shape[1], sigma=radii[0], sigmaN=radii[-1], learning_rate=alphas[0],
                   learning_rateN=alphas[-1], random_seed=seed, xp=np, n_parallel=batch_size)

    # Add the codes as weights to the SOM, this could be from a previous MST run.
    pysom._weights = codes.reshape((xdim, xdim, data.shape[1]))
    pysom.train(data, rlen, verbose=True)

    codes = pysom.get_weights()

    return codes.reshape((ncodes, data.shape[1]))


def calculate_numbaSOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf=eucl, seed=None):
    """Calculate the SOM using the numbaSOM implementation.

    This code comes from the public GitHub repository https://github.com/nmarincic/numbasom by user nmarincic.
    The parameters are the same as the SOM function in this file.
    """
    if seed is not None:
        np.random.seed(seed)
    # find the dimensions of the data
    xdim = int(np.sqrt(ncodes))
    n = data.shape[0]
    # calculate the number of iterations
    niter = rlen * n

    # create the numbaSOM object with a given size. most of the time, a 10x10 grid is used.
    numbasom = numbaSOM(som_size=(xdim, xdim), is_torus=False)

    lattice = numbasom.train(data, niter)

    # the lattice has to be reshaped to the shape (samples, dimensions)
    codes = lattice.reshape((ncodes, data.shape[1]))
    return codes


@jit(nopython=True, parallel=True)
def calculate_originalSOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf=eucl, seed=None):
    if seed is not None:
        np.random.seed(seed)
    xdists = np.zeros(ncodes)
    n = data.shape[0]
    px = data.shape[1]
    niter = rlen * n
    threshold = radii[0]
    thresholdStep = (radii[0] - radii[1]) / niter
    change = 1.0

    for k in range(niter):
        if k % n == 0:
            if change < 1:
                k = niter
            change = 0.0

        i = np.random.randint(n)

        nearest = 0
        for cd in range(ncodes):
            xdists[cd] = distf(data[i, :], codes[cd, :])
            if xdists[cd] < xdists[nearest]:
                nearest = cd

        if threshold < 1.0:
            threshold = 0.5
        alpha = alphas[0] - (alphas[0] - alphas[1]) * k / niter

        for cd in range(ncodes):
            if nhbrdist[cd, nearest] > threshold:
                continue

            for j in range(px):
                tmp = data[i, j] - codes[cd, j]
                change += abs(tmp)
                codes[cd, j] += tmp * alpha

        threshold -= thresholdStep
    return codes


@jit(nopython=True, parallel=True)
def calculate_SOM_batch(data, codes, nhbrdist, alphas, radii, ncodes, rlen, batch_size=1, distf=eucl, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n = data.shape[0]
    px = data.shape[1]
    batches = n // batch_size
    niter = rlen * batches
    threshold = radii[0]
    thresholdStep = (radii[0] - radii[1]) / niter
    start_lrate = alphas[0]
    end_lrate = alphas[1]

    for k in range(niter):
        # Randomly select a batch of data points
        batch_indices = np.random.choice(n, batch_size, replace=False)
        batch_data = data[batch_indices]

        bmu_counts = np.zeros(ncodes)
        updates = np.zeros_like(codes)

        # Find the BMU for each data point in the batch and accumulate updates
        for i in prange(batch_size):
            # Find the nearest code vector (BMU)
            xdists = np.zeros(ncodes)
            for cd in range(ncodes):
                xdists[cd] = distf(batch_data[i, :], codes[cd, :])

            nearest = np.argmin(xdists)

            # Accumulate updates for each code vector
            for cd in prange(ncodes):
                if nhbrdist[cd, nearest] > threshold:
                    continue

                bmu_counts[cd] += 1
                for j in range(px):
                    updates[cd, j] += (batch_data[i, j] - codes[cd, j])

        if threshold < 1.0:
            threshold = 0.5

        # Compute the learning rate using linear annealing
        alpha = start_lrate - (start_lrate - end_lrate) * k / niter

        # Apply accumulated updates to the code vectors
        for cd in range(ncodes):
            if bmu_counts[cd] > 0:
                for j in range(px):
                    updates[cd, j] /= bmu_counts[cd]
                    codes[cd, j] += alpha * updates[cd, j]

        threshold -= thresholdStep

    return codes


@jit(nopython=True, parallel=True)
def lr_SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf=eucl, seed=None):
    if seed is not None:
        np.random.seed(seed)
    xdists = np.zeros(ncodes)
    n = data.shape[0]
    px = data.shape[1]
    niter = rlen * n
    threshold = radii[0]
    thresholdStep = (radii[0] - radii[1]) / niter
    change = 1.0
    start_lrate = alphas[0]
    end_lrate = alphas[1]

    for k in range(niter):
        if k % n == 0:
            if change < 1:
                k = niter
            change = 0.0

        i = np.random.randint(n)

        nearest = 0
        for cd in range(ncodes):
            xdists[cd] = distf(data[i, :], codes[cd, :])
            if xdists[cd] < xdists[nearest]:
                nearest = cd

        if threshold < 1.0:
            threshold = 0.5

        # use cosine annealing for the learning rate
        alpha = end_lrate + 0.5 * (start_lrate - end_lrate) * (1 + np.cos(np.pi * k / niter))
        for cd in range(ncodes):
            if nhbrdist[cd, nearest] > threshold:
                continue

            for j in range(px):
                tmp = data[i, j] - codes[cd, j]
                change += abs(tmp)
                codes[cd, j] += tmp * alpha

        threshold -= thresholdStep
    return codes


@jit(nopython=True, parallel=True)
def map_data_to_codes(data, codes, distf=eucl):
    n_codes = codes.shape[0]
    nd = data.shape[0]
    nn_codes = np.zeros(nd)
    nn_dists = np.zeros(nd)
    for i in range(nd):
        minid = -1
        mindist = np.inf
        for cd in range(n_codes):
            tmp = distf(data[i, :], codes[cd, :])
            if tmp < mindist:
                mindist = tmp
                minid = cd
        nn_codes[i] = minid
        nn_dists[i] = mindist
    return nn_codes, nn_dists
