"""fish_data module contains the helper functions for the model build of the
Nature Conservancy Fisheries Kaggle Competition.

Dependencies:
    * numpy as np
    * os
    * scipy.ndimage as ndimage
    * scipy.misc as misc

"""

#dependencies
import numpy as np
import os
from scipy import ndimage, misc

def mutate_image(image) :
    """Receives an image array and returns an image array with a random set of
        distortions.
    Distortions include:
        * flip horizontally (0.5)
        * flip vertically (0.5)
        * slight distortion of coloring (0.5 for any distortion at all, plus
            random float for amount of distortion)
        * random horizontal shift (image is cropped vertically between 1/5-1/15
            the image. random top or bottom is cropped.)
        * random vertical shift (image is cropped horizontally between 1/10 to
            1/20 of the image.  random left or right is cropped.) """
    assert len(image.shape) == 3 , 'Image is not in 3D'
    assert image.shape[2] == 3, 'Image is not in RGB format'

    flip_hor, flip_ver, sigma, vert_off, hor_off, top, left = [np.random.randint(0,2),
                                                    np.random.randint(0,2),
                                                    np.random.choice([0, np.sqrt(np.random.random())]),
                                                    np.random.randint(5,15),
                                                    np.random.randint(10,20),
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
    return(ndimage.filters.gaussian_filter(image_prime, sigma))

def standardize(image, std_y, std_x) :
    """Normalizes and resizes an image array to a standard height, length, and
    pixel range."""
    assert len(image.shape) is 3, "Image array is not in 3-dimensions"

    image_n = (image.astype(float) - 255.0 / 2) / 255.0 # pixel depth of RGB is 255.0.  Line takes image array to mean == 0
    image_std = misc.imresize(image_n, size = (std_y, std_x))
    return np.expand_dims(image_std, 0) #create 4th dimension.  Will be concatenated in this first dimension for the batch size




def generate_epoch_set_list_and_label_array(min_each) :
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

        #make one-hot label array
        fish_label_arr = np.zeros([min_each, len(os.listdir('data/train')[1:])])
        fish_label_arr[:, i] = 1

        #add to the master list / master array
        try :
            master_file_names = master_file_names + fish_file_names
            master_label_arr = np.concatenate([master_label_arr,fish_label_arr], 0)
        except :
            master_file_names = fish_file_names
            master_label_arr = fish_label_arr

    print("\nTests")
    print("     Master list of filenames contains 8 * min_each filenames: {}".format(len(master_file_names)==16000))
    print("     Label is assigned only once per row entry: {}".format(
        np.all(np.sum(master_label_arr,1) == np.ones((8*min_each)))))
    print("     There are 'min_each' labels for each fish column: {}".format(np.all(np.sum(master_label_arr,0) == np.full(8, min_each))))
    return master_file_names, master_label_arr

def make_batch(filename_list, offset, batch_size, std_y, std_x) :
    """Iterates through a filename list to load an RGB image of any pixel
    dimensions, mutate the image using the `mutate_image` function, normalize
    the pixel values and pixel dimensions using the `standardize` function, and
    return an array for concatenation into a 4D array.  The function can be used
    to assemble training batches or to bring the validation array into the
    kernel environment."""

    for filename in filename_list[offset : (offset+batch_size)] :
        #download the array
        image = ndimage.imread(filename)
        assert len(image.shape) == 3, 'Image is not in 3 dimensions'
        assert (image.shape[2]) == 3, 'Image is not in RGB format'
        image_mut = mutate_image(image)
        image_norm = standardize(image_mut, std_y, std_x)
        try :
            arr = np.vstack([arr,image_norm])
        except :
            arr = image_norm
    assert arr.shape == (batch_size, std_y, std_x, 3), "ERROR: array ({}) is not of correct dimensions".format(arr.shape)
    return arr

def make_label(label_arr, offset, batch_size) :
    """Returns the label associated with a training batch generation.  Fn is
    necessary for navigating the ends of the epoch list."""
    return label_arr[offset:(offset+batch_size), :]

def count_nodes(std_y, std_x, pool_steps, final_depth ) :
    """Calculates the number of flattened nodes after a number of 'VALID' pool
    steps of strides = [1,2,2,1]"""
    y = std_y
    x = std_x
    for _ in range(pool_steps) :
        y = (y // 2)
        x = (x // 2)
    return y*x*final_depth
