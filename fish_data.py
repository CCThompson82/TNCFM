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


def retrieve_fovea(key, fovea_dictionary) :
    """
    Retrieves a fovea from the fovea_dictionary.
    """
    fov_dict = fovea_dictionary.get(key)
    scale = fov_dict.get('scale')
    f = fov_dict.get('f')
    y_off = fov_dict.get('y_offset')
    x_off = fov_dict.get('x_offset')

    fov = misc.imresize(
            misc.imload(f, mode = 'RGB'),
            size = scale, mode = 'RGB')[y_off:(y_0ff+fov_size), x_off:(x_off+fov_size), :]
    return fov



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
    if pixel_norm == 'standard' :
        fovea = fovea.astype(np.float32)
        fovea = (fovea / 255.0) - 0.5
    elif pixel_norm == 'float' :
        fovea = fovea.astype(np.float32)
        fovea = (fovea / 256.0)
        fovea = np.clip(fovea, a_min = 0.0, a_max = 1.0)
    elif pixel_norm == 'centre' :
        fovea = fovea.astype(np.float32)
        fovea = (fovea - 128.0)  # TODO : use sklearn to set mean equal to zero?
    else :
        pass

    return fovea


def generate_fovea(f, fov_size = 224, scale = [2.0, 1.0, 0.5], y_bins = 10, x_bins = 10, pixel_norm = 'standard') :
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

        arr = np.zeros([(y_bins*x_bins), fov_size, fov_size, num_channels], dtype= np.uint8)
        counter = 0
        for y in y_locs :
            for x in x_locs :
                fov = np.expand_dims(img[y:(y+fov_size), x:(x+fov_size), :], 0)
                if pixel_norm is not None :
                    fov = process_fovea(fov, pixel_norm = pixel_norm )
                arr[counter, :, :, :] = fov
                counter +=1
        arr_list.append(arr)

    return arr_list



def annote_fovea_manager(f, image_dictionary, fovea_dictionary, validation_dictionary,
            fov_size = 224, scale = [2.0, 1.0, 0.5], y_bins = 10, x_bins = 10, pixel_norm = 'standard', verdir = None) :
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

    hr_lab = image_dictionary.get('train').get(f).get('image_label' )
    print("Label:",hr_lab)
    nof_stage = {}
    valid_stage = {}
    fish_stage = {}
    for s in scale :

        print("Scale: {}".format(s))
        img = misc.imresize(misc.imread(f, mode = 'RGB'), size = s, mode = 'RGB')

        y_max, x_max, d = img.shape

        y_locs = stride_hack(y_max, kernel = fov_size, bins = y_bins)
        x_locs = stride_hack(x_max, kernel = fov_size, bins = x_bins)

        arr = np.zeros([(y_bins*x_bins), fov_size, fov_size, num_channels], dtype= np.uint8)
        counter = 0
        for y in y_locs :
            for x in x_locs :
                fov = np.expand_dims(img[y:(y+fov_size), x:(x+fov_size), :], 0)
                if pixel_norm is not None :
                    fov = process_fovea(fov, pixel_norm = pixel_norm )
                arr[counter, :, :, :] = fov
                counter +=1
        assert x_bins > y_bins, "ERROR in image representation"
        fig, axarr = plt.subplots(y_bins, x_bins, figsize = (x_bins*2.5,y_bins*2.5), sharex=True, sharey=True)
        for i in range(arr.shape[0]) : # counter value?
            y = i // x_bins
            x = i % x_bins
            axarr[y,x].imshow(arr[i, :, :, :])
            axarr[y,x].set_ylabel(i)
            axarr[y,x].set_yticks([])
            axarr[y,x].set_xticks([])
        fig.subplots_adjust(hspace = 0.01)
        plt.show()



        nof_dict = {}
        fish_dict = {}

        for i in range(arr.shape[0]) :
            nof_dict[i] = {'f' : f, 'scale' : s, 'y_offset' : y_locs[i // x_bins], 'x_offset' : x_locs[i % x_bins], 'fov_label' : 'NoF' }

        if hr_lab == 'NoF' :
            pass
        else :
            fish_list = input('Stage to fish    ')
            fi_list = fish_list.split(',')

            if fi_list != ['']:
                for fi in fi_list :
                    nof_dict[int(fi.strip())]['fov_label'] = image_dictionary.get('train').get(f).get('image_label') #change the fovea_label based on the image_label
                    fish_dict[f+'_'+str(np.random.random())] = nof_dict.pop(int(fi.strip()))

        fish_stage.update(fish_dict)
        for n in nof_dict :
            nof_stage.update({ nof_dict.get(n).get('f')+'_'+str(np.random.random()) : nof_dict.get(n) } )




    master_commit = input("commit to version directory master fovea or validation dictionary?  (y/n/v)     ")
    if master_commit == 'y' :

        print("Updated Fovea Training Set")
        fovea_dictionary.get(hr_lab).update(fish_stage)
        fovea_dictionary.get('NoF').update(nof_stage)
        for fish_class in fovea_dictionary :
            print("{} : {} images".format(fish_class, len(fovea_dictionary.get(fish_class))))

        with open(verdir+'/fovea_dictionary.pickle', 'wb') as ffd :
            pickle.dump(fovea_dictionary, ffd)
    if master_commit == 'v' :

        print("Updated Validation Training Set")
        validation_dictionary.get(hr_lab).update(fish_stage)
        if hr_lab == 'NoF' :
            validation_dictionary.get('NoF').update(nof_stage)
        for fish_class in validation_dictionary :
            print("{} : {} images".format(fish_class, len(validation_dictionary.get(fish_class))))
        with open(verdir+'/validation_dictionary.pickle', 'wb') as ffd :
            pickle.dump(validation_dictionary, ffd)
