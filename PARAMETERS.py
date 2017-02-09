"""This is a file containing the parameters necessary for GRAPH.py and SESSION.py
run for the Nature Conservancy Fisheries Kaggle Competition"""

num_epochs = 30
#Preprocessing
std_sizes = [256, 256]
crop_size = 224

# General
num_channels = 3
num_labels = 8
batch_size = 25
stddev = 0.01

# convolution
kernel_sizes = [11, 5, 3, 3, 3]
conv_depths = [96, 256, 384, 384, 256]
conv_strides = [4, 1, 1, 1, 1]

pool_strides = [2, 2, 2]
pool_kernels = [3, 3, 3]

final_depth = conv_depths[-1]

#dropout
kp = 0.5

# fully connected
fc1_depth = 4096
fc2_depth = 2048

#regularization
beta = 1e-1

# Learning rate
init_rate = 1e-2
per_steps = len(files_train)*10
decay_rate = 0.1

# Momentum
momentum = 0.9
#momentum_decay = 0.9995

# Report rate
validate_interval = 10


# Report for ipynb wrapper
y, x = crop_size, crop_size
d1 = y*x*num_channels
print("Dimensions for each entry: {}x{}x{} = {}".format(y, x, num_channels, d1))

y,x,d,d2 = fd.count_nodes(y, x, kernel_sizes[0], conv_strides[0], conv_depths[0])
y,x,d, d2p = fd.count_nodes(y,x, pool_kernels[0], pool_strides[0], conv_depths[0], pad='VALID')
print("Dimensions after first convolution step (with max pool): {}x{}x{} = {}".format(y, x, d, d2p))

y,x,d,d3 = fd.count_nodes(y,x, kernel_sizes[1], conv_strides[1], conv_depths[1])
y,x,d, d3p = fd.count_nodes(y,x, pool_kernels[1], pool_strides[1], conv_depths[1], pad='VALID')
print("Dimensions after second convolution step (with max pool): {}x{}x{} = {}".format(y, x, d, d3p))

y,x,d,d4 = fd.count_nodes(y,x, kernel_sizes[2], conv_strides[2], conv_depths[2])
print("Dimensions after third convolution step: {}x{}x{} = {}".format(y, x, d, d4))

y,x,d,d5 = fd.count_nodes(y,x, kernel_sizes[3], conv_strides[3], conv_depths[3])
print("Dimensions after fourth convolution step: {}x{}x{} = {}".format(y, x, d, d5))

y,x,d,d6 = fd.count_nodes(y,x, kernel_sizes[4], conv_strides[4], conv_depths[4])
y,x,d, d6p = fd.count_nodes(y,x, pool_kernels[2], pool_strides[2], conv_depths[4], pad='VALID')
print("Dimensions after fifth convolution step (with max pool): {}x{}x{} = {}".format(y, x, d, d6p))

nodes_exit_convolution = d6p

print("Dimensions after first connected layer: {}".format(fc1_depth))
print("Dimensions after second connected layer: {}".format(fc2_depth))
print("Final dimensions for classification: {}".format(num_labels))
