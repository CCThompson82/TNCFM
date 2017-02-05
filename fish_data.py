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



def mutate_image(image) :
    """Receives an image array and returns an image array with a random set of
        distortions.
    Distortions include:
        * flip horizontally (p = 0.5)
        * flip vertically (p = 0.5)
        * slight distortion of coloring (random sigma for each channel - see
            shift_colour fn in this module)
        * random horizontal shift (image is cropped vertically between 1/15 to
            1/20 the image. top or bottom is randomly chosen for chop.)
        * random vertical shift (image is cropped horizontally between 1/15 to
            1/20 of the image.  left or right is randomly chosen for chop.) """
    assert len(image.shape) == 3 , 'Image is not in 3D'
    assert image.shape[2] == 3, 'Image is not in RGB format'

    flip_hor, flip_ver, sigma, vert_off, hor_off, top, left = [np.random.randint(0,2),
                                                    np.random.randint(0,2),
                                                    np.random.normal(0, 0.5, size = 3),
                                                    np.random.randint(15,20),
                                                    np.random.randint(15,20),
                                                    np.random.randint(0,2),
                                                    np.random.randint(0,2)]
    #shift the image
    if top > 0 :
        if left > 0 :
            image_prime = image[image.shape[0]//vert_off:, image.shape[1]//hor_off:, :]
        else :
            image_prime = image[image.shape[0]//vert_off:, :-image.shape[1]//hor_off, :]
    else :
        if left > 0 :
            image_prime = image[:-image.shape[0]//vert_off, image.shape[1]//hor_off:, :]
        else :
            image_prime = image[:-image.shape[0]//vert_off, :-image.shape[1]//hor_off, :]

    if flip_hor > 0 :
        image_prime = np.fliplr(image_prime)

    if flip_ver > 0 :
        image_prime = np.flipud(image_prime)

    image_prime = shift_colour(image_prime, channel = 0, sigma = sigma[0] )
    image_prime = shift_colour(image_prime, channel = 1, sigma = sigma[1] )
    image_prime = shift_colour(image_prime, channel = 2, sigma = sigma[2] )
    return image_prime

def shift_colour(image, channel, sigma) :
    """ Shifts the colour of an RGB image by sigma"""
    assert len(image.shape) == 3, 'Image not in RGB format'

    new_image = image.copy()
    panel = new_image[:,:,channel].astype(float)
    # normalize and clip saturated pixels to avoid infinity terms
    panel = (panel / 255.0).clip(min = 1e-4, max = 0.9999)
    # convert pixel values to logits
    panel = special.logit(panel)
    # add sigma value to logit values
    panel = panel + sigma
    # convert back to sigmoid
    panel = special.expit(panel)
    # convert to pixel depth of 255.0
    panel = panel*255.0
    # cover up old panel with colour shifted panel
    new_image[:,:,channel] = panel
    return new_image

def standardize(image, std_y, std_x, normalize = True) :
    """Normalizes and resizes an image array to a standard height, length, and
    pixel range."""
    assert len(image.shape) is 3, "Image array is not in 3-dimensions"
    image_std = misc.imresize(image, size = (std_y, std_x))
    if normalize :
        image_n = (image_std.astype(np.float32) - 255.0 / 2) / 255.0 # pixel depth of RGB is 255.0.  Line takes image array to mean == 0
    else :
        image_n = image_std.astype(np.float32)
    return np.expand_dims(image_n, 0) #create 4th dimension.  Will be concatenated in this first dimension for the batch size



def generate_filenames_list() :
    """Iterates through the 'data/train' folders of the working directory to
    generate a list of filenames"""
    for i, species_ID in enumerate(os.listdir('data/train')[1:]) :
        fish_file_names = []
        fish_file_names = ['data/train/'+species_ID+'/'+x for x in os.listdir('data/train/'+species_ID) ]
        fish_count = len(fish_file_names)

        try :
            master_file_names = master_file_names + fish_file_names
        except :
            master_file_names = fish_file_names
    return master_file_names


def generate_balanced_epoch(min_each, shuffle = True) :
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

def make_batch(filename_list, offset, batch_size, std_y, std_x, normalize = True, mutate = True) :
    """Iterates through a filename list to load an RGB image of any pixel
    dimensions, mutate the image using the `mutate_image` function, normalize
    the pixel values and pixel dimensions using the `standardize` function, and
    return an array for concatenation into a 4D array.  The function can be used
    to assemble training batches or to bring the validation array into the
    kernel environment."""

    entry_counter = batch_size
    index = offset
    while entry_counter > 0 :
        try :
            image = ndimage.imread(filename_list[index])
        except : # tripped if index exceeds the index length of filename_list
            index = 0
            image = ndimage.imread(filename_list[index])

        assert len(image.shape) == 3, 'Image is not in 3 dimensions'
        assert (image.shape[2]) == 3, 'Image is not in RGB format'

        if mutate :
            image_mut = mutate_image(image)
        else :
            image_mut = image

        image_norm = standardize(image_mut, std_y, std_x, normalize)

        if entry_counter == batch_size :
            arr = image_norm
        else :
            arr = np.vstack([arr, image_norm])

        entry_counter -= 1
        index += 1

        if index == len(filename_list) :
            index = 0

    assert arr.shape == (batch_size, std_y, std_x, 3), "ERROR: array <{}> is not of correct dimensions: <{}>".format(arr.shape, (batch_size, std_y, std_x, 3))
    return arr



def make_label(filename_list, offset, batch_size) :
    """Returns the label associated with a training batch generation.  Fn also
    navigates the ends of the epoch list."""
    entry_counter = batch_size
    index = offset
    while entry_counter > 0 :
        file = filename_list[index]
        start = file.find('train/') + 6
        end = file.find('/img')
        try :
            label_arr = np.vstack([label_arr,
                                  np.array([ file[start:end] == x for x in ['ALB','BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']]).astype(int)
                                    ])
        except :
            label_arr = np.array([ file[start:end] == x for x in ['ALB','BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']]).astype(int)

        entry_counter -= 1
        index += 1
        if index == len(filename_list) :
            index = 0

    return label_arr

def count_nodes(std_y, std_x, pool_steps, final_depth ) :
    """Calculates the number of flattened nodes after a number of 'VALID' pool
    steps of strides = [1,2,2,1]"""
    y = (std_y // 3) + 1
    x = (std_x // 3) + 1
    for _ in range(pool_steps) :
        y = (y // 2)
        x = (x // 2)
    return y*x*final_depth

def show_panel(image) :
    plt.figure(figsize=(15,30))
    plt.subplot(1,4,1)
    plt.imshow(image[:,:,0])
    plt.subplot(1,4,2)
    plt.imshow(image[:,:,1])
    plt.subplot(1,4,3)
    plt.imshow(image[:,:,2])
    plt.subplot(1,4,4)
    plt.imshow(image)
    plt.show()
