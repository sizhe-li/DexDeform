import numpy as np

# credit: https://github.com/NVlabs/ACID/blob/bbf0c49195daf77a92d88d2f3238c0f4b8b0be0e/ACID/src/data/transforms.py

# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.
    It adds noise to point cloud data.
    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.
        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.
    It subsamples the point cloud data.
    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.
        Args:
            data (dict): data dictionary
        '''
        indices = np.random.randint(data.shape[0], size=self.N)

        return data[indices]
