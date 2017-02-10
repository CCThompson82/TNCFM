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


def generate_balanced_filenames_epoch(f_list, labels, shuffle = True) :
    """
    Returns a shuffled list of filenames, of which some will be duplicates, such
    that each fish class is represented equally, along with corresponding one-hot
    labels for the list.
    """
    assert len(list) == labels.shape[0]
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
        scalar = len(fish_list) // mAx
        remain = len(fish_list) % mAx
        new_fish_list = fish_list * scalar + np.random.choice(fish_list, remain)
        new_master_list += new_fish_list

    if shuffle :
        np.random.shuffle(new_master_list)

    # Generate labels for the new set
    new_labels = make_labels(new_master_list)

    return new_master_list, new_labels



def process_batch(  f_list, labels, offset, batch_size,
                    std_size, crop_size, crop_mode = 'centre',
                    pixel_offsets = [155.0, 155.0, 155.0], mutation = False) :
    """
    Fn preprocesses a batch of images and collects associated labels for input
    into a tensorflow graph placeholder.
    """
    assert crop_mode in ['centre', 'random', 'many']
    if batch_size == 'all' :
        batch_size = len(f_list)
    if offst is None :
        offset = 0

    for i in range(offset, offset+batch_size) :
        #read image into environment
        try :
            img = misc.imread(f_list[i])
        except :
            i = i - len(f_list)
            img = misc.imread(f_list[i])
        #shape of img
        y, x, d = img.shape
        # determine short side
        if y <= x :
            y_short = True
        else :
            x_short = True
        #resize the short size the std_size
        if y_short == True :
            img = mix.imresize(img, size = (std_size, std_size*(x//y), 3))
        elif x_short == True :
            img = misc.imresize(img, size = (std_size*(y//x), std_size, 3))
        print("Intermediate shape: {}".format(img.shape))

        #crop image to crop size
        y, x, d = img.shape
        if crop_mode == 'centre' :
            y_off = (y-crop_size) // 2
            x_off = (x - crop_size) // 2
        elif crop_mode == 'random' :
            y_off = np.random.randint(0,(y-crop_size),1)
            x_off = np.random.randint(0, (x-crop_size), 1)
        elif crop_mode == 'all' :
            pass
        else :
            return "ERROR in image crop"

        img = img[ y_off:(y_off+crop_size), x_off:(x_off+crop_size), d ]

        # centre pixel colours
        img[:,:,0] = img.astype(np.float32) - pixel_offsets[0]
        img[:,:,1] = img.astype(np.float32) - pixel_offsets[1]
        img[:,:,2] = img.astype(np.float32) - pixel_offsets[2]

        if mutation :
            if np.random.int(0,2,1) == 1 :
                img = np.fliplr(img)
            if np.random.int(0,2,1) == 1 :
                img = np.flipud(img)
            if np.random.int(0,2,1) == 1 :
                img = np.rot90(img)

        if i == offset : # trips on first iteration
            batch = np.expand_dims(img, 0)
            batch_label = labels[[i], :]
        else :
            batch = np.concatenate([batch, np.expand_dims(img, 0)], 0)
            batch_label = np.concatenate([batch_label, labels[[i], :]])
    return batch, batch_labels
