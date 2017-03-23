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
    * pickle

"""

#dependencies
import numpy as np
import os
from scipy import ndimage, misc, special
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

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




def show_panel(image) :
    """Convenience function for showing an inline montage of the colour and merged channels"""
    plt.figure(figsize=(16,8))
    plt.subplot(1,4,1)
    plt.imshow(image[:,:,0], cmap = 'Reds')
    plt.subplot(1,4,2)
    plt.imshow(image[:,:,1], cmap = 'Greens')
    plt.subplot(1,4,3)
    plt.imshow(image[:,:,2], cmap = 'Blues')
    plt.subplot(1,4,4)
    plt.imshow(image)
    plt.show()




def process_fovea(fovea, pixel_norm = 'standard', mutation = False) :
    """
    Fn preprocesses a single fovea array.

    If mutation == True, modifications to input images will be made, each with 0.5
    probability:

        * smallest dimension resized to standard height and width supplied in size param
        * each channel centered to mean near zero.  Deviation is not normalized.
        * if mutate == True :
            * random flip left right
            * random flip up down
            * random rotation 90 degrees
            * TODO : random colour adjustment

    Pixel value normalization is under development.

    """

    if mutation :
        if np.random.randint(0,2,1) == 1 :
            fovea = np.fliplr(fovea)
        if np.random.randint(0,2,1) == 1 :
            fovea = np.flipud(fovea)
        if np.random.randint(0,2,1) == 1 :
            fovea = np.rot90(fovea)


    #pixel normalization
    fovea = fovea.astype(np.float32)
    if pixel_norm == 'standard' :
        fovea = (fovea / 255.0) - 0.5
    elif pixel_norm == 'float' :
        fovea = (fovea / 256.0)
        fovea = np.clip(fovea, a_min = 0.0, a_max = 1.0)
    elif pixel_norm == 'centre' :
        fovea = (fovea - 128.0)  # TODO : use sklearn to set mean equal to zero?
    else :
        pass

    return fovea


def format_pred_arr(f, fov_size = 224, scale = [2.0, 1.0, 0.5], y_bins = 10, x_bins = 10) :
    """
    Converts a high-resolution RGB image into an array stack for fovea prediction
    in the FISHFINDER model.
    """
    global num_channels
    num_channels = 3

    def stride_hack(length, kernel, bins) :
        """
        Calculates the kernel start locations for a definied number of bins with valid padding.
        """
        a_max = length - kernel
        if bins == 'all' :
            bins = a_max
        base_stride = a_max // (bins - 1)
        stride_remainder = a_max % (bins - 1)
        #assert stride_remainder < bins, 'ERROR'
        hack = np.ones([stride_remainder]).tolist()
        hack += np.zeros([bins - len(hack)]).tolist()

        cursor = 0
        ret_vec = []
        while len(ret_vec) < bins :
            ret_vec.append(cursor)
            jitter = int(hack.pop(np.random.randint(0,len(hack))))
            cursor += base_stride + jitter
        return ret_vec

    arr_list = []
    for s in scale :
        img = misc.imresize(misc.imread(f, mode = 'RGB'), size = s, mode = 'RGB')

        y_max, x_max, d = img.shape

        y_locs = stride_hack(y_max, kernel = fov_size, bins = y_bins)
        x_locs = stride_hack(x_max, kernel = fov_size, bins = x_bins)

        arr = np.zeros([(y_bins*x_bins), fov_size, fov_size, num_channels])
        counter = 0
        for y in y_locs :
            for x in x_locs :
                fov = np.expand_dims(img[y:(y+fov_size), x:(x+fov_size), :], 0)
                fov_s = process_fovea(fov)
                arr[counter, :, :, :] = fov_s

                counter +=1
        arr_list.append(arr)

    return arr_list
