from __future__ import print_function

import os

import numpy as np
from scipy import misc


def load_test_data(phone, dped_dir, IMAGE_SIZE):
    test_directory_phone = dped_dir + str(phone) + '/test_data/patches/' + str(phone) + '/'
    test_directory_dslr = dped_dir + str(phone) + '/test_data/patches/canon/'

    NUM_TEST_IMAGES = len([name for name in os.listdir(test_directory_phone)
                           if os.path.isfile(os.path.join(test_directory_phone, name))])

    test_data = np.zeros((NUM_TEST_IMAGES, IMAGE_SIZE))
    test_answ = np.zeros((NUM_TEST_IMAGES, IMAGE_SIZE))

    for i in range(0, NUM_TEST_IMAGES):

        I = np.asarray(misc.imread(test_directory_phone + str(i) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        test_data[i, :] = I

        I = np.asarray(misc.imread(test_directory_dslr + str(i) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        test_answ[i, :] = I

        if i % 100 == 0:
            print(str(round(i * 100 / NUM_TEST_IMAGES)) + "% done", end="\r")

    return test_data, test_answ


def load_batch(phone, dped_dir, TRAIN_SIZE, IMAGE_SIZE):
    disk = False

    train_directory_phone = dped_dir + str(phone) + '/training_data/' + str(phone) + '/'
    train_directory_dslr = dped_dir + str(phone) + '/training_data/canon/'

    disk_mat_phone = dped_dir + str(phone) + '/training_data/' + str(phone) + '_{}_{}.npy'.format(TRAIN_SIZE,
                                                                                                  IMAGE_SIZE)
    disk_mat_dslr = dped_dir + str(phone) + '/training_data/canon_{}_{}.npy'.format(TRAIN_SIZE, IMAGE_SIZE)

    if disk and os.path.isfile(disk_mat_phone) and os.path.isfile(disk_mat_dslr):
        print("loading npy")
        train_data = np.load(disk_mat_phone)
        train_answ = np.load(disk_mat_dslr)
        print("loaded")
        return train_data, train_answ

    # if TRAIN_SIZE == -1 then load all images
    NUM_TRAINING_IMAGES = len([name for name in os.listdir(train_directory_phone)
                               if os.path.isfile(os.path.join(train_directory_phone, name))])

    if TRAIN_SIZE == -1:

        TRAIN_SIZE = NUM_TRAINING_IMAGES
        TRAIN_IMAGES = np.arange(0, TRAIN_SIZE)
    else:
        TRAIN_IMAGES = np.random.choice(np.arange(0, NUM_TRAINING_IMAGES), TRAIN_SIZE, replace=False)

    train_data = np.zeros((TRAIN_SIZE, IMAGE_SIZE))
    train_answ = np.zeros((TRAIN_SIZE, IMAGE_SIZE))

    i = 0
    for img in TRAIN_IMAGES:

        I = np.asarray(misc.imread(train_directory_phone + str(img) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        train_data[i, :] = I

        I = np.asarray(misc.imread(train_directory_dslr + str(img) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        train_answ[i, :] = I

        i += 1
        if i % 100 == 0:
            print(str(round(i * 100 / TRAIN_SIZE)) + "% done", end="\r")

    if disk:
        print("saving npy")
        np.save(disk_mat_phone, train_data)
        np.save(disk_mat_dslr, train_answ)
        print("saved")

    return train_data, train_answ
