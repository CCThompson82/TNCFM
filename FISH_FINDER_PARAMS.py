# General
num_channels = 3
fov_size = 224
num_labels = 8
batch_size = 64
num_fovea = 100
summary_rate = 10 # tensorboard writing events after x batches
stddev = 0.1

label_dict = {'ALB' : np.array([1,0,0,0,0,0,0,0]),
               'BET' : np.array([0,1,0,0,0,0,0,0]),
               'DOL' : np.array([0,0,1,0,0,0,0,0]),
               'LAG' : np.array([0,0,0,1,0,0,0,0]),
               'NoF' : np.array([0,0,0,0,1,0,0,0]),
               'OTHER' : np.array([0,0,0,0,0,1,0,0]),
               'SHARK' : np.array([0,0,0,0,0,0,1,0]),
               'YFT' : np.array([0,0,0,0,0,0,0,1]) }

# Convolutions
conv_kernel = 3
conv_stride = 1

conv_depth = [32, 32, 64, 64, 128, 128, 256, 256, 512, 512]

# Pooling
pool_kernel = 3
pool_stride = 2

# Dense layers
nodes_after_conv = 18432

fc_depth = [2048, 1024, 512]
drop_prob = 0.5 # TODO refactor to a more accurate term of keep prob
