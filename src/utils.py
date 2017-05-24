import numpy as np


def rand_unit_sphere(npoints, ndim=1000):
    '''
    Generates "npoints" number of vectors of size "ndim"
    such that each vectors is a point on an "ndim" dimensional sphere
    that is, so that each vector is of distance 1 from the center

    npoints -- number of feature vectors to generate
    ndim -- how many features per vector

    returns -- np array of shape (npoints, ndim), dtype=float64
    '''
    vec = np.random.randn(npoints, ndim)
    vec = np.divide(vec, np.expand_dims(np.linalg.norm(vec, axis=1), axis=1))
    return vec


def shuffle_assigned_noises(noises):
    '''
    shuffles all of the noises assigned to images
    done evety N epoch to avoid plateaus
    '''
    new_noises = {}
    keys = noises.keys()
    values = list(noises.values())
    np.random.shuffle(values)

    for k, v in zip(keys, values):
        new_noises[k] = v

    return new_noises
