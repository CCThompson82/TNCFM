# General
num_channels = 3
fov_size = 128
num_labels = 8
batch_size = 64

stddev = 0.2

label_dict = {'ALB' : np.array([1,0,0,0,0,0,0,0]),
               'BET' : np.array([0,1,0,0,0,0,0,0]),
               'DOL' : np.array([0,0,1,0,0,0,0,0]),
               'LAG' : np.array([0,0,0,1,0,0,0,0]),
               'NoF' : np.array([0,0,0,0,1,0,0,0]),
               'OTHER' : np.array([0,0,0,0,0,1,0,0]),
               'SHARK' : np.array([0,0,0,0,0,0,1,0]),
               'YFT' : np.array([0,0,0,0,0,0,0,1]) }

reverse_label_dict = { 0 : 'ALB',
                       1 : 'BET',
                       2 : 'DOL',
                       3 : 'LAG',
                       4 : 'NoF',
                       5 : 'OTHER',
                       6 : 'SHARK',
                       7 : 'YFT'}

# Convolutions
conv_kernel = 3
conv_stride = 1

pretrained_path = '../../PreTrained_Models/VGG_19/variables/'
conv_depth = [64, 64,
              128, 128,
              256, 256, 256, 256,
              512, 512, 512, 512]
# Pooling
pool_kernel = 2
pool_stride = 2

# Dense layers
nodes_after_conv = 8192

fc_depth = [4096, 2048, 512, 128]
keep_prob = [0.6, 0.7, 0.8, 0.9]


# Representations
pred_scales = [0.75, 1.0, 1.5]
bins_y = 12
bins_x = 16
pred_batch = 64
assert ((bins_y*bins_x) % pred_batch) == 0, 'pred batch is not a factor of the fovea number'
