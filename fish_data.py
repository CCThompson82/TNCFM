"""fish_data module contains the helper functions for the model build of the
Nature Conservancy Fisheries Kaggle Competition.

Dependencies:
    * numpy as np
    * scipy.ndimage as ndimage
    * scipy.misc as misc

"""

#dependencies
import numpy as np
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
            image_prime = test_img[image.shape[0]//vert_off:, test_img.shape[1]//hor_off:, :]
        else :
            image_prime = test_img[image.shape[0]//vert_off:, :-test_img.shape[1]//hor_off, :]
    else :
        if left > 0 :
            image_prime = test_img[:-image.shape[0]//vert_off, test_img.shape[1]//hor_off:, :]
        else :
            image_prime = test_img[:-image.shape[0]//vert_off, :-test_img.shape[1]//hor_off, :]

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
    image_std = misc.imresize(image_n, std_y, std_x)
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
    return master_file_names, master_label_arr
