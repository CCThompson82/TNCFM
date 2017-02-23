"""Parameters for AUTOENCODER_GRAPH.py"""


#General parameters
num_channels = 3
fovea_presize = 224 # Must match master_dict[filename][parameters][pixel_dims]
fovea_size = 224 # required by session call in batch processing
assert fovea_size <= fovea_presize
stddev = 0.2
batch_size = 16*5 # equal to the number of fovea taken from each image multiplied by the number of high resolution images per batch


autoencoder_depths = [8, 8]
autoencoder_kernels = [5, 3]
autoencoder_strides = [2, 2]
