#/usr/bin/python3
# -*- coding: utf-8 -*-

# library modules
import os
import pickle
import logging
import time
import json

from random import randint
from random import choice
from queue import Queue
from math import ceil
from threading import Thread
from threading import Lock
from logs import get_logger

# external library modules
from PIL import Image
import numpy as np

def img2PIL(image):
    """
    Converts an image to a pillow object.
    If it is greyscale image, then convert it to RGB first

    :param image: path to the image file
    """
    img = Image.open(image)

    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.load()

    return img

def img2np(image, size=None):
    """
    Converts an image to numpy data.
    If it is greyscale image, then convert it to RGB first
    and then change to numpy array

    :param image: path to the image file
    :param size: If given reshape the image to this size.
    """
    img = Image.open(image)

    if img.mode != 'RGB':
        img = img.convert('RGB')
    if size:
        img = img.resize(size)
    img.load()

    return np.asarray(img, dtype = "int32")

def imgs2np(function):
    """
    Convert list of pillow images to its numpy equivalent.
    """

    def wrapper(*args, **kwargs):
        images = function(*args, **kwargs)

        for i, image in enumerate(images):
            images[i] = np.asarray(image, dtype=np.int32)

        return images

    return wrapper

def gen_mean_activity(base_dir):
    """
    Generate mean activity for each channel over entire training set

    :param base_dir: Base directory for training
    """
    logger = get_logger('Mean Activity', 'mean.log')
    RGB = np.zeros((3,))
    lock = Lock()
    def mean_activity_folder(base_dir):
        _RGB = np.zeros((3,))
        logger.info("Starting directory: %s", base_dir)
        for image in os.listdir(base_dir):
            img = Image.open(os.path.join(base_dir,
                                          image))
            img = resize(img)

            npimg = np.array(img)
            _RGB += npimg.mean(axis=(0,1))

        with lock:
            nonlocal RGB
            RGB += _RGB

        logger.info("Ending directory: %s", base_dir)

    count = 0
    threads = []
    for i, folder in enumerate(os.listdir(os.path.join(base_dir))):
        folder_path = os.path.join(base_dir, folder)
        count += len(os.listdir(folder_path))
        thread = Thread(target=mean_activity_folder,
                        args=(folder_path,))
        thread.start()
        threads.append(thread)
        if i % 100 == 0:
            for t in threads:
                t.join()
            threads = []

    for t in threads:
        t.join()

    logger.info("RGB: %s, count: %d", str(RGB), count)
    RGB /= count

    with open('mean.pkl', 'wb') as handle:
        pickle.dump(RGB, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_mean_activity():
    """
    Get mean activity for each channel(Red, Gree, Blue)
    """
    with open('mean.pkl', 'rb') as handle:
        return pickle.load(handle)

def resize(img):
    """
    Resize the image isotropically with smallest image side as S.

    Let S be the smallest side of an isotropically-rescaled image.
    To rescale an image isotropically we maintain the aspect ratio(ratio
    of width to height and vice versa).
    """
    new_smallest_side = randint(256, 512)

    # Rescale isotropically
    if img.width <= img.height:
        img = img.resize((new_smallest_side,
                          round(new_smallest_side * img.height / img.width)))
    else:
        img = img.resize((round(new_smallest_side * img.width / img.height),
                          new_smallest_side))

    # Change the mode to RGB if it is not
    if img.mode != 'RGB':
        img = img.convert('RGB')

    return img

def preprocess(function, resize_crop=True, augment=True):
    """
    Preprocess (resize, crop, optionally augment) the image
    """
    def crop(image, image_size):
        """
        Randomly crop `image_size` of the the `image`
        """
        _width, _height = (image.size[0] - image_size[0],
                           image.size[1] - image_size[1])

        start_width, start_height = (randint(0, _width),
                                     randint(0, _height))

        return image.crop((start_width, start_height,
                           start_width + image_size[0],
                           start_height + image_size[1]))

    def _augment(image, transpose=False, color_shift=False):
        """
        Do random horizontal flipping and random RGB color shift.

        With 50% probability flip the image horizontally.
        With 90% probability add a quantity to the image pixels as
        described in `AlexNet <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_
        paper.

        The eigen value and eigen vector for imagenet is taken from
        `stackoverflow <https://stackoverflow.com/questions/43328600/does-anyone-have-the-eigenvalue-and-eigenvectors-for-alexnets-pca-noise-from-th>`_

        :param image: Doing transpose on pillow object is easier. But
                      doing color shift on numpy is easier. so the type of
                      image is dependent what exactly is being performed.
        """
        if transpose and randint(1, 2) == 1:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if color_shift and randint(1, 10) != 1:
            eigval = np.array([[55.46], [4.794], [1.148]])
            eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                               [-0.5808, -0.0045, -0.8140],
                               [-0.5836, -0.6948, 0.4203]])
            alpha = np.random.normal(loc=0, scale=0.1, size=(3, 1))
            pca = np.dot(eigvec, alpha * eigval)

            image = image + pca.reshape((1, 1, 3))

        return image

    def wrapper(*args, **kwargs):
        self = args[0]

        img = function(*args, **kwargs)
        if resize_crop:
            img = resize(img)
            img = crop(img, self.image_size)
        # flip the image horizontally
        if augment:
            img = _augment(img, transpose=True)
        img.load()

        npimg = np.asarray(img, dtype = "int32")
        # add multiples of pca to the image
        if augment:
            npimg = _augment(npimg, color_shift=True)

        # Subtract mean activity from each channel
        mean = get_mean_activity()

        return npimg - mean.reshape((1, 1, 3))

    return wrapper

class Store:
    """
    A store to keep batches of data for deep learning
    using threading.
    """
    def __init__(self, source, max_qsize):
        """
        :param source: It will tell how to get the data.
          This is a tuple of function, total size of data for one epoch,
          and batch size. The function will be used to get the data
          to store
        :type source: tuple ==> (function, int, int)
        :param max_qsize: Maximum number of batches it can store
        """
        self.function, self.data_size, self.batch_size = source
        self.max_qsize = max_qsize
        self.queue = Queue(max_qsize)
        self.logger = logging.getLogger('VGG.utils.Store')

    def _write(self, i):
        """
        Helper function to pass to the thread class to read data parallelly
        """
        X, Y = self.function(i)
        self.queue.put((X, Y))
        self.logger.debug("The batch no %d is stored", i)

    def write(self):
        """
        Store datas by using the function given using threading.

        It should read datas from disk parallelly.
        """
        threads = []
        for idx in range(ceil(self.data_size / self.batch_size)):
            while len(threads) >= self.max_qsize:
                for i, t in enumerate(threads):
                    if not t.is_alive():
                        del threads[i]
                        break
                if len(threads) < self.max_qsize: break
                time.sleep(.5)
            thread = Thread(target=self._write, args=(idx,))
            # don't need to read batches if the main program exits
            thread.daemon = True
            thread.start()
            threads.append(thread)

    def read(self):
        """
        Generator to read data from the store.

        Creates a generator to read data from store(not disk).
        It first starts reading the data from disk using threading
        so that the data in the store is always available while reading.
        """
        thread = Thread(target=self.write)
        thread.daemon = True
        thread.start()
        for _ in range(ceil(self.data_size / self.batch_size)):
            yield self.queue.get()

def read_vgg_conf():
    """
    Read the configurations from vgg19.json
    """
    with open('vgg19.json') as f:
        vgg_conf = json.load(f)
    return vgg_conf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar = 'image-path',
                        help = 'ImageNet train dataset path')
    args = parser.parse_args()

    train_path = os.path.join(args.image_path)
    random_train_folder = choice(os.listdir(train_path))
    folder_path = os.path.join(train_path, random_train_folder)
    random_image = choice(os.listdir(folder_path))
    image_path = os.path.join(folder_path, random_image)

    print("Image path", image_path)
    print("Image shape", img2np(image_path).shape)
    if not os.path.exists('mean.pkl'):
        gen_mean_activity(args.image_path)
    print("Mean activity", get_mean_activity())
