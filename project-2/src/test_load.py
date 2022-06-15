import numpy as np

data = np.load('./data/samples/projections/mustache_directions.npz')

print(list(data.keys()))
codes = data['w']

print(codes)
print(codes.shape)