"""fish_data module contains the helper functions for the model build of the
Nature Conservancy Fisheries Kaggle Competition.

Dependencies:
    * numpy as np
    * os
    * scipy.ndimage as ndimage
    * scipy.misc as misc
    * scipy.special as special
    * matplotlib.pyplot as plt
    * tensorflow as tf

"""

#dependencies
import numpy as np
import os
from scipy import ndimage, misc, special
import matplotlib.pyplot as plt
import tensorflow as tf

def generate_filenames_list(subdirectory = 'data/train/', subfolders = True) :
    """Iterates through the default 'data/train' folders of the working directory to
    generate a list of filenames"""
    if subfolders :
        for i, species_ID in enumerate(os.listdir(subdirectory)[1:]) :
            fish_file_names = []
            fish_file_names = [subdirectory+species_ID+'/'+x for x in os.listdir(subdirectory+'/'+species_ID) ]
            fish_count = len(fish_file_names)

            try :
                master_file_names = master_file_names + fish_file_names
            except :
                master_file_names = fish_file_names
    else :
        master_file_names = [subdirectory+x for x in os.listdir(subdirectory)]
    return master_file_names

def generate_balanced_filenames_epoch(min_each, shuffle = True) :
    """Function to generate a list of filenames to be used for each training epoch
    with a corresponding label array.  Most file names will be used  multiple  times
    in order that each fish is drawn into a training batch an equivalent number of
    times."""
    # Count the images in each set and append to a fish_list
    for i, species_ID in enumerate(os.listdir('data/train')[1:]) :
        fish_file_names = []
        fish_file_names = ['data/train/'+species_ID+'/'+x for x in os.listdir('data/train/'+species_ID) ]
        fish_count = len(fish_file_names)
        assert min_each > fish_count, 'Listed minimum number of images is exceeded by the actual number of images.  Increase minimum number to generate a balanced dataset'
        #tack on multiples of the original fish_file_names and then randomly select a handful of file names to fill out the remainder up to the min_each value
        multiples = min_each // fish_count
        remainder = min_each % fish_count
        fish_file_names = (fish_file_names* multiples) + np.random.choice(fish_file_names, remainder).tolist()
        print("'{}' set contains {} filenames from which to sample".format(species_ID, len(fish_file_names)))

        #add to the master list / master array
        try :
            master_file_names = master_file_names + fish_file_names
        except :
            master_file_names = fish_file_names
    print("{} filenames are in the training set list".format(len(master_file_names)))
    t = master_file_names[0:5]
    if shuffle :
        np.random.shuffle(master_file_names)
        print("Filename list is shuffled: {}".format(t != master_file_names[0:5]))
    return master_file_names

def make_labels(filename_list, directory_string = 'train/', end_string = '/img') :
    """Receives a list of filenames and returns an ordered one-hot label
    array by finding the fish species ID within the filename string."""

    for file in filename_list :
        start = file.find(directory_string) + 6
        end = file.find(end_string)
        try :
            label_arr = np.vstack([label_arr,
                                  np.array([ file[start:end] == x for x in ['ALB','BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']]).astype(int)
                                    ])
        except :
            # inititate label_arr if it is the first record entry
            label_arr = np.array([ file[start:end] == x for x in ['ALB','BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']]).astype(int)

    return label_arr

def count_nodes(y_in, x_in, conv_depths, conv_strides, pool_strides ) :
    """Calculates the number of total nodes present in the last layer of a
    convolution plus max_pooling architecture.  Calculations assume that
    convolution is 'SAME' padded, and pooling is 'VALID' padded."""
    y = y_in
    x = x_in
    for i in range(len(conv_depths)) :
        if conv_strides[i] > 1 :
            y = (y // conv_strides[i]) + 1
            x = (x // conv_strides[i]) + 1
    for i in range(len(pool_strides)) :
        y = (y // pool_strides[i])
        x = (x // pool_strides[i])

    return y*x*conv_depths[-1]

def decode_image(image_read, size, num_channels = 3,
                                    mutate = False, brightness_delta = None,
                                                    contrast_limits = None,
                                                    hue_delta = None,
                                                    saturation_limits = None ) :
    """Converts a dequeued image read from filename to a single tensor array,
    with modifications:
        * resized to standard height and width supplied in size param
        * normalized to mean near zero, min == -0.5, max ==  0.5
        * distortions if mutate == True :
            * random flip left right
            * random flip up down
            * # TODO : random crop and resize to standard size ???
            * random brightness
            * random contrast
            * random hue
            * random saturation

    providing distortion if mutate == True"""
    assert (len(size) == 2), 'Size does not contain height and width values'
    img = tf.image.decode_jpeg(image_read, channels = num_channels )
    img = tf.image.resize_images(img, size = size)
    img = (tf.to_float(img) * (1.0 / 255.0)) - 0.5

    if mutate :
        # TODO : set assertions based on delta performances in training later, can't have images unrecognizable, e.g.
        """
        assert  brightness_delta < 1, 'brightness_delta not in range [0,1)''
        assert  contrast_delta < 1, 'contrast_delta is not in range [0,1)'
        assert  hue_delta <= 0,5, 'hue_delta is not in range of [0,0.5]'
        assert  saturation_delta, 'Saturation factor is not '
        """
        img = tf.image.random_brightness(img, max_delta = brightness_delta)
        img = tf.image.random_contrast(img, lower = contrast_limits[0], upper = contrast_limits[1])
        img = tf.image.random_hue(img, max_delta = hue_delta)
        img = tf.image.random_saturation(img, lower = saturation_limits[0], upper = saturation_limits[1])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
    return img

def show_panel(image) :
    """Convenience function for showing an inline montage of the colour and merged channels"""
    plt.figure(figsize=(10,30))
    plt.subplot(2,2,1)
    plt.imshow(image[:,:,0], cmap = 'Reds')
    plt.subplot(2,2,2)
    plt.imshow(image[:,:,1], cmap = 'Greens')
    plt.subplot(2,2,3)
    plt.imshow(image[:,:,2], cmap = 'Blues')
    plt.subplot(2,2,4)
    plt.imshow(image)
    plt.show()
