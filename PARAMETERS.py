"""This is a file containing the parameters necessary for GRAPH.py and SESSION.py
run for the Nature Conservancy Fisheries Kaggle Competition"""

num_epochs = 5
#Preprocessing
std_y = 300
std_x = 500

# General
num_channels = 3
num_labels = 8
batch_size = 25
stddev = 0.2

# convolution
kernel_sizes = [5, 3, 3, 3, 3, 3]
conv_depths = [64, 64, 128, 256, 512, 256] # NOTE : first 64 currently not used with dilation2d
conv_strides = [4, 1, 1, 1, 1, 1]

pool_strides = [2, 2, 2, 2]

final_depth = conv_depths[-1]

#dropout
kp = 0.75

# fully connected
fc1_depth = 256
fc2_depth = 64

#regularization
beta = 1e-1

# Learning rate
init_rate = 5e-2
per_steps = len(files_train)
decay_rate = 0.1

# Report rate
validate_interval = 10
