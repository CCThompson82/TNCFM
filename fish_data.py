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

def make_coordinates_dict(filename_list, resize_val = 1.0, presize = 256, bins = (4, 3), force = False, store = False) :
    """
    Utilizes a nested dictionary to crop images into multiple fovea for generation of a naive (i.e. unlabeled)
    image set.

    """
    try :
        print("Attempting to load dictionary pickle...")
        with open('coordinates_dictionary.pickle', 'rb') as handle:
            master_dict = pickle.load(handle)
        print("Dictionary loaded!")
    except :
        print("Not able to load dictionary!")
        force = True

    if force :
        print("Generating coordinates...")
        assert len(bins) == 2, "bins needs y and x dimension"
        assert 0 < resize_val <= 1.0, "resize_val is not a float between 0 and 1"

        master_dict = {}
        for f in filename_list :
             # TODO : assert fd.make_labels receives a list in fish_data.py
            img_dict = {f: {'label': make_labels([f])}}

            img = misc.imresize(misc.imread(f, mode = 'RGB'), size = resize_val, mode = 'RGB')
            y, x, _ = img.shape

            max_y = y - presize
            max_x = x - presize

            stride_y = (y // bins[0]) - ((presize - (y // bins[0])) // (bins[0]-1))
            stride_x = (x // bins[1]) -  ((presize - (x // bins[1])) // (bins[1]-1))

            v = 0
            coord_list = []
            for _ in range(bins[0]) :
                h = 0
                for _ in range(bins[1]) :
                    #hack to avoid overshooting the end of the image by a single pixel
                    while h+presize > x :
                        h -= 1
                    while v+presize > y :
                        v -= 1
                    coord_list.append((v, h))
                    h += stride_x
                v += stride_y

            img_dict[f]['parameters'] = {'prescaled' : resize_val, 'pixel_dims' : presize}
            img_dict[f]['fovea_offsets'] = coord_list

            master_dict.update(img_dict)

        if store :
            with open('coordinates_dictionary.pickle', 'wb') as fp:
                pickle.dump(master_dict, fp)

    return master_dict

def count_nodes(x, y, kernel, stride, conv_depth, pad = 'SAME') :
    """Calculates the number of total nodes present in the next layer of a
    convolution OR max_pooling event."""
    assert pad in ['SAME', 'VALID']

    if pad == 'SAME' :
        pad = kernel // 2
    else :
        pad = 0
    xp = (x - kernel + (2*pad)) // stride + 1
    yp = (y - kernel + (2*pad)) // stride + 1

    return xp, yp, conv_depth, xp*yp*conv_depth

def decode_image(image_name, size, num_channels = 3, mean_channel_vals = [155.0, 155.0, 155.0], mutate = False, crop = 'random', crop_size = 224) :
    """Converts a dequeued image read from filename to a single tensor array,
    with modifications:
        * smallest dimension resized to standard height and width supplied in size param
        * each channel centered to mean near zero.  Deviation is not normalized.
        * if mutate == True :
            * random flip left right
            * random flip up down
            * TODO : random colour adjustment
            * random crop from standard size to crop size (e.g. 256x256 to 224x224)
    """
    assert (len(size) == 2), 'Size does not contain height and width values'
    assert crop in ['random', 'centre', 'all'], "Crop must set to be 'random', 'centre', or 'all'"
    #read image into memory
    img = tf.image.decode_jpeg(image_name, channels = num_channels )

    # TODO : refactor this to maintain actual aspect ratio, not just maintain the average aspect ratio.  Could be done with tf.cond statements
    new_x = np.int(size[1] * (1250 / 730))
    #set smallest dimension (y) to size[0]
    img = tf.image.resize_images(img, size = [size[0], new_x])
    img = tf.image.crop_to_bounding_box(img, offset_height = 0, offset_width = ((new_x - size[1]) // 2), target_height = size[0], target_width = size[1])
    #crop based on parameter
    if crop == 'random' :
        h = np.random.randint(0,(size[0]-crop_size), 1).astype(np.int32)[0]
        w = np.random.randint(0,(size[1]-crop_size), 1).astype(np.int32)[0]
    elif crop == 'centre' :
        h = tf.to_int32((size[0] - crop_size) // 2)
        w = tf.to_int32((size[1] - crop_size) // 2)
    elif crop == 'all' :
        pass
    else :
        return "ERROR in image crop"
    img = tf.image.crop_to_bounding_box(img, offset_height = h, offset_width = w, target_height = crop_size, target_width = crop_size)

    # centre each color channel
    for c in range(3) :
        img = tf.to_float(img)
        img = tf.subtract(img, mean_channel_vals)

    if mutate :
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
    return img

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


def generate_balanced_filenames_epoch(f_list, labels, shuffle = True) :
    """
    Returns a shuffled list of filenames, of which some will be duplicates, such
    that each fish class is represented equally, along with corresponding one-hot
    labels for the list.
    """
    assert len(f_list) == labels.shape[0]
    assert labels.shape[1] == 8

    #Count the number of images with each fish class using the labels

    print("Fish counts: {}".format(np.sum(labels, 0)))
    mAx = np.max(np.sum(labels,0))

    #Collect filenames based on fish class
    master_list = []
    for i in range(labels.shape[1]) :
        fish_list = []
        for ix, f in enumerate(f_list) :
            if labels[ix, i] == 1 :
                fish_list.append(f)
        master_list.append(fish_list)


    # Duplicate filenames as needed to balance the set
    new_master_list = []
    for fish_list in master_list :
        scalar = mAx // len(fish_list)
        remain = mAx % len(fish_list)
        new_fish_list = fish_list * scalar + np.random.choice(fish_list, remain).tolist()
        new_master_list = new_master_list + new_fish_list

    if shuffle :
        np.random.shuffle(new_master_list)

    # Generate labels for the new set
    new_labels = make_labels(new_master_list)
    print("New fish counts: {}".format(np.sum(new_labels,0)))

    return new_master_list, new_labels



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


def prepare_batch(dictionary, set_list, batch_size, fov_size, label_dictionary, return_label = 'onehot') :
    """
    Retrieves fovea from a dictionary that contains filname, coordinates of
    fovea, fovea_label, pre-scale float.  As fovea are added to the batch, they
    are removed from training_set_list to avoid duplicate use.
    """
    # TODO assert that dictionary contains filename ('f') and 'coordinates' keys
    def retrieve_fovea(key, dictionary = dictionary, fov_size = fov_size, label_dictionary = label_dictionary) :
        """Convenience function to retrive the fovea array with one-hot label"""
        f_dict = dictionary.get(key)
        f = f_dict['f']
        scale = f_dict['scale']
        y_off, x_off = f_dict['coordinates']['y_offset'], f_dict['coordinates']['x_offset']
        fovea = misc.imresize(
                        misc.imread(f, mode = 'RGB'),
                            size = scale,
                            mode = 'RGB')[y_off:(y_off+fov_size), x_off:(x_off+fov_size), :]
        fovea = np.expand_dims(process_fovea(fovea, pixel_norm = 'standard', mutation = True), 0)
        label = np.expand_dims(label_dictionary.get(f_dict['fovea_label']),0) # TODO : refactor so that label return is callable.  Label not necessary for stage fovea call

        return fovea, label, key

    fovea, label, key = retrieve_fovea(
                            set_list.pop(np.random.randint(0, len(set_list))))
    X_batch = fovea
    if return_label == 'onehot' :
        y_batch = label
    keys = [key]

    while X_batch.shape[0] != batch_size :
        fovea, label, key = retrieve_fovea(
                        set_list.pop(
                            np.random.randint(0, len(set_list))))

        X_batch = np.concatenate([X_batch, fovea], 0)
        if return_label == 'onehot' :
            y_batch = np.concatenate([y_batch,label], 0)
        keys.append(key)
    if return_label == 'onehot' :
        return X_batch, y_batch, keys
    else :
        return X_batch, keys

def fovea_generation(image_dictionary, num_fovea = 100, fov_size = 224, scale_min = 0.4, scale_max = 1.0) :
    """
    Function for random sampling of high-resolution image files, followed by
    random fovea generation.
    """
    new_fovea_dict = {}
    f_list = [x for x in image_dictionary]
    samples_f_list = np.random.choice(f_list, num_fovea).tolist()

    def random_float_from_range(a_min, a_max) :
        """Convenience function for the sampling of a uniform distribution of a specific range."""
        z = a_max - a_min
        return z * np.random.rand() + a_min

    while len(samples_f_list) > 0 :
        f = samples_f_list.pop(np.random.randint(0,len(samples_f_list)))
        scale = random_float_from_range(a_min = scale_min, a_max = scale_max)
        shape = misc.imresize(misc.imread(f, mode = 'RGB'), size = scale, mode = 'RGB').shape
        y_offset = np.random.randint(0, shape[0]-fov_size, 1)[0]
        x_offset = np.random.randint(0, shape[1]-fov_size, 1)[0]

        new_fovea_dict[f] = {'f' : f ,
                             'scale': scale,
                             'coordinates' : {'y_offset' : y_offset, 'x_offset' : x_offset},
                             'image_label' : image_dictionary[f]['image_label'],
                             'staged_steps' : 0,
                             'fovea_label' : None }
    return new_fovea_dict


def stage_set_supervisor(stgd_lgts, staged_dictionary, training_set_dictionary, keys, label_dict, reverse_label_dict) :
    """
    This function manages the dictionaries that contain the training and
    staged sets of fovea.  After random fovea are added to the staged set, and run
    through the current FISHFINDER model for label prediction, this fn will
    destage, keep, or commit the fovea to the training_set_dictionary.

    Each high-resolution image that contains a fish is composed of fovea that
    are either depict NoF or depict the fish. Because labels exist for all of the
    training set of high-resolution images, this function will utilize the predicted
    probabilities that the fovea either contains no fish, and thus the fovea label is
    set to 'NoF', or contains some type of fish, in which case the image label would be
    propigated to the fovea label.

    A series of conditions will ensure that fovea are not incorrectly labeled and
    committed automatically to the growing training set.

    """
    for i, key in enumerate(keys) :
        pred = stgd_lgts[i,:]
        keep_threshold = float(open('keep_threshold.txt', 'r').read().strip())
        commit_threshold = open('commit_threshold.txt', 'r').read().strip()

        fish_prob = np.sum(pred) - pred[4]
        nof_prob = pred[4]

        if fish_prob < keep_threshold and nof_prob < keep_threshold :
            keep = False
        else : # high enough confidence prediction has been made

            # Is the predicted fovea label possible?
            #   * NoF is always possible
            #   * any fish is not a possible fovea label for NoF image label
            #   * TEST images can have any label

            NoF_bool = np.argmax(pred) == 4

            # cannot predict a fish in 'NoF' images
            if fish_prob >= keep_threshold :
                fish_bool = (staged_dictionary[key].get('image_label') != 'NoF') # fish was predicted and image label is NoF ;
            else :
                fish_bool = False # fish was predicted but image label was NoF.  not a legitimate prediction

            # keep all high confidence prediction fovea from the TEST set

            test_bool = staged_dictionary[key].get('image_label') == 'TEST'

            if np.any([NoF_bool, fish_bool, test_bool]) :
                keep = True
            else :
                keep = False

        if keep :
            if test_bool :
                staged_dictionary[key]['fovea_label'] = reverse_label_dict.get(np.argmax(pred))
            # if the image label is known and fovea is predicted as a fish, propigate the image label as the fovea label
            elif fish_bool  :
                staged_dictionary[key]['fovea_label'] = staged_dictionary[key].get('image_label')
            elif NoF_bool :
                staged_dictionary[key]['fovea_label'] = 'NoF'

            staged_dictionary.get(key)['staged_steps'] += 1

            if commit_threshold == 'Manual' :
                commit = False
            elif staged_dictionary.get(key)['staged_steps'] == commit_threshold :
                commit = True
            else :
                commit = False

            if commit :
                training_set_dictionary.append( {key : staged_dictionary.pop(key)} )
        else :
            _ = staged_dictionary.pop(key)
    return staged_dictionary, training_set_dictionary

def manual_stage_manager(staged_dictionary, training_set_dictionary, fovea_size, stage_step_threshold, md) :
    """
    Convience function that prompts the user to verify fovea labels predicted by the FISHFINDER model for the
    fovea that are currently staged.  The stage_step_threshold can be used to filter only those fovea where
    predictions are stable, having passed the staging test `n` or more consecutive times.  The user may commit as
    labeled, or may change the fovea label to the fish class when NoF has been incorrectly predicted.  Finally, the
    user may also destage the fovea if the fovea prediction is incorrect or ambiguous.
    """

    keys = [key for key in staged_dictionary]
    correct_count, missed_fish, missed_NoF, ambig_count = 0, 0, 0, 0
    for key in keys :
        fov_dict = staged_dictionary.get(key)
        if fov_dict['staged_steps'] >= stage_step_threshold :
            print("="*50)
            print("Image Label: {}     Fovea prediction: {}".format(fov_dict.get('image_label'), fov_dict.get('fovea_label')))

            scale = fov_dict.get('scale')
            y_off = int(fov_dict.get('coordinates').get('y_offset'))
            x_off = int(fov_dict.get('coordinates').get('x_offset'))
            fov = (misc.imresize(
                        misc.imread(
                            fov_dict.get('f')),
                            size = scale,
                            mode = 'RGB')[y_off:(y_off+fovea_size),
                                                              x_off:(x_off+fovea_size), :])
            show_panel(fov)

            if (fov_dict.get('image_label') == 'NoF' and
                fov_dict.get('fovea_label') == 'NoF') :
                print("Auto commit triggered!")
                commit = 'c'
            elif (fov_dict.get('image_label') == 'TEST') :
                commit = input("Correct, commit : c; Incorrect, overide with 'NoF' : n, Incorrect, destage: d       Answer: ")
            else :
                commit = input("Correct, commit : c; Incorrect, overide as fish: f, Incorrect, overide as 'NoF': n,  Incorrect, destage: d       Answer: ")

            if commit == 'c' :
                new_key = key+';_yx_'+str(y_off)+'_'+str(x_off)
                dict_to_add = {new_key: fov_dict}
                training_set_dictionary.update(dict_to_add)
                _ = staged_dictionary.pop(key)
                correct_count += 1
            elif commit == 'f' :
                fov_dict['fovea_label'] = fov_dict['image_label']
                new_key = key+';_yx_'+str(y_off)+'_'+str(x_off)
                dict_to_add = {new_key: fov_dict}
                training_set_dictionary.update(dict_to_add)
                _ = staged_dictionary.pop(key)
                missed_fish += 1
            elif commit == 'n' :
                fov_dict['fovea_label'] = 'NoF'
                new_key = key+';_yx_'+str(y_off)+'_'+str(x_off)
                dict_to_add = {new_key: fov_dict}
                training_set_dictionary.update(dict_to_add)
                _ = staged_dictionary.pop(key)
                missed_NoF += 1

            elif commit == 'd' :
                _ = staged_dictionary.pop(key)
                ambig_count += 1

            else :
                pass
    print("\n\nNew size of training_set_dictionary: {}".format(len(training_set_dictionary)))
    print("New size of staged_set_dictionary: {}".format(len(staged_dictionary)))

    print("\nBatch Accuracy: {}".format(correct_count / np.sum([correct_count, missed_fish, missed_NoF])))


def format_pred_arr(f, fov_size = 224, scale = [1.5, 1.0, 0.5], y_bins = 10, x_bins = 10) :
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
