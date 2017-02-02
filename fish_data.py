"""fish_data module contains the helper functions for the model build of the
Nature Conservancy Fisheries Kaggle Competition.

Dependencies:
    * numpy as np
    * os
    * scipy.ndimage as ndimage
    * scipy.misc as misc
    * scipy.special as special
    * matplotlib.pyplot as plt

# TODO : Make a test for mutation in make_batch to ensure observed overfitting
# is due to model parameters and not to me simply having had repeated each image
# over and over in the training set. """

#dependencies
import numpy as np
import os
from scipy import ndimage, misc, special
import matplotlib.pyplot as plt

def mutate_image(image) :
    """Receives an image array and returns an image array with a random set of
        distortions.
    Distortions include:
        * flip horizontally (0.5)
        * flip vertically (0.5)
        * slight distortion of coloring (0.5 for any distortion at all, plus
            random float for amount of distortion)
        * random horizontal shift (image is cropped vertically between 1/10 to
            1/15 the image. top or bottom is cropped.)
        * random vertical shift (image is cropped horizontally between 1/15 to
            1/20 of the image.  left or right is cropped.) """
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

def standardize(image, std_y, std_x) :
    """Normalizes and resizes an image array to a standard height, length, and
    pixel range."""
    assert len(image.shape) is 3, "Image array is not in 3-dimensions"
    image_std = misc.imresize(image, size = (std_y, std_x))
    image_n = (image_std.astype(np.float32) - 255.0 / 2) / 255.0 # pixel depth of RGB is 255.0.  Line takes image array to mean == 0
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


def balance_data_and_label_array(species_ID, min_each) :
    """Function to generate a list of filenames to be used for each training epoch
    with a corresponding label array.  Most file names will be used  multiple  times
    in order that each fish is drawn into a training batch an equivalent number of
    times.  Return from this function must be appended / concatenated into a master list
    / array, respectively."""

    fish_file_names = []
    fish_file_names = ['data/train/'+species_ID+'/'+x for x in os.listdir('data/train/'+species_ID) ]
    fish_count = len(fish_file_names)
    assert min_each > fish_count, 'Listed minimum number of images is exceeded by the actual number of images.  Increase minimum number to generate a balanced dataset'
    #tack on multiples of the original fish_file_names and then randomly select a handful of file names to fill out the remainder up to the min_each value
    multiples = min_each // fish_count
    remainder = min_each % fish_count
    fish_file_names = (fish_file_names* multiples) + np.random.choice(fish_file_names, remainder).tolist()
    print("'{}' set contains {} filenames from which to sample".format(species_ID, len(fish_file_names)))

    #make one-hot label array
    fish_label_arr = np.zeros([min_each, len(os.listdir('data/train')[1:])])
    fish_label_arr[:, i] = 1

    return fish_file_names, fish_label_arr



def make_batch(filename_list, offset, batch_size, std_y, std_x, standardize = True,  mutate = True) :
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

        if standardize :
            image_norm = standardize(image_mut, std_y, std_x)
        else :
            image_norm = image_mut

        if entry_counter == batch_size :
            arr = image_norm
        else :
            arr = np.vstack([arr, image_norm])

        entry_counter -= 1
        index += 1

    assert arr.shape == (batch_size, std_y, std_x, 3), "ERROR: array <{}> is not of correct dimensions: <{}>".format(arr.shape, (batch_size, std_y, std_x, 3))
    return arr



def make_label(label_arr, offset, batch_size) :
    """Returns the label associated with a training batch generation.  Fn is
    necessary for navigating the ends of the epoch list."""
    entry_counter = batch_size
    index = offset
    while entry_counter > 0 :
        try :
            entry = label_arr[index, :]
        except : # tripped if index exceeds the index length of filename_list
            index = 0
            image = label_arr[index, :]

        if entry_counter == batch_size :
            arr = entry
        else :
            arr = np.vstack([arr, entry])
        entry_counter -= 1
        index += 1
    assert arr.shape == (batch_size, 8), "ERROR in label retrieval"
    return arr

def count_nodes(std_y, std_x, pool_steps, final_depth ) :
    """Calculates the number of flattened nodes after a number of 'VALID' pool
    steps of strides = [1,2,2,1]"""
    y = std_y
    x = std_x
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
