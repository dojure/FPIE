# python test_model.py model=iphone_orig dped_dir=dped/ test_subset=full iteration=all resolution=orig use_gpu=true

import os
import sys

import numpy as np
import tensorflow as tf
from scipy import misc

import models
import utils

# process command arguments
print(sys.argv)
model, dped_dir, test_subset, iteration, resolution, use_gpu, run, kernel_size, depth, blocks, parametric, s_conv, convdeconv = utils.process_test_model_args(
    sys.argv)

dirname = model + "_" + run
if run == "":
    dirname = model

if model.endswith("_orig"):
    phone = model.replace("_orig", "")
    orig = True
    if phone == "iphone":
        dirname = "iphone_convdeconv16_iteration_40000"
    else:
        raise ValueError("No pre-trained model exists for '{}'".format(phone))
else:
    phone = model
    orig = False

# get all available image resolutions
res_sizes = utils.get_resolutions()

# get the specified image resolution
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = utils.get_specified_res(res_sizes, phone, resolution)

# disable gpu if specified
config = tf.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None

# create placeholders for input images
x_ = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

# generate enhanced image
if convdeconv:
    enhanced = models.convdeconv(x_image, depth, parametric=parametric, s_conv=s_conv)
else:
    if orig:
        enhanced = models.convdeconv(x_image, 16, parametric=False, s_conv=False)
    else:
        enhanced = models.resnet(x_image, kernel_size, depth, blocks, parametric=parametric, s_conv=s_conv)

with tf.Session(config=config) as sess:
    print("phone:", phone)
    print("dirname:", dirname)

    test_dir = dped_dir + phone + "/test_data/full_size_test_images/"
    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

    name_subset = []

    if test_subset == "small":
        # use five first images only
        test_photos = test_photos[0:5]
    elif test_subset == "all" or test_subset == "full":
        pass
    elif test_subset.startswith("["):
        import ast

        name_subset = ast.literal_eval(iteration)
    else:
        name_subset = [int(test_subset)]

    if name_subset:
        collect = []
        for i in name_subset:
            for f in test_photos:
                if str(i) + ".jpg" == f:
                    collect.append(f)
        test_photos = collect

    if not os.path.exists("visual_results"):
        os.makedirs("visual_results")

    if orig:

        # load pre-trained model
        saver = tf.train.Saver()
        saver.restore(sess, "models_orig/" + dirname + ".ckpt")

        for photo in test_photos:
            # load training image and crop it if necessary

            print("Testing original " + phone + " model, processing image " + photo)
            image = np.float16(misc.imresize(misc.imread(test_dir + photo), res_sizes[phone])) / 255

            image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
            image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])

            # get enhanced image

            enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
            enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

            before_after = np.hstack((image_crop, enhanced_image))
            photo_name = photo.rsplit(".", 1)[0]

            # save the results as .png images

            misc.imsave("visual_results/" + model + "_" + photo_name + "_enhanced.png", enhanced_image)
            misc.imsave("visual_results/" + model + "_" + photo_name + "_before_after.png", before_after)

    else:

        num_saved_models = int(len([f for f in os.listdir("models/") if f.startswith(dirname + "_iteration")]) / 2)

        if iteration == "all" or iteration == "last":
            files = os.listdir("models")
            import re

            regex = re.compile(r'\d+')
            iters = [int(regex.findall(filename)[-1]) for filename in files if
                     dirname + "_" in filename and ".ckpt.index" in filename]
            if iteration == "last":
                iteration = [max(iters)]
            else:
                iteration = sorted(iters)
        elif "-" in iteration:
            its = iteration.split("-")
            if not its[1]:
                its[1] = num_saved_models + 1
            iteration = np.arange(int(its[0]), int(its[1])) * 1000
        elif iteration.startswith("["):
            import ast

            iteration = ast.literal_eval(iteration)
        else:
            iteration = [int(iteration)]
        print(iteration)
        for i in iteration:

            # load pre-trained model
            saver = tf.train.Saver()
            saver.restore(sess, "models/" + dirname + "_iteration_" + str(i) + ".ckpt")

            for photo in test_photos:
                # load training image and crop it if necessary

                print("iteration " + str(i) + ", processing image " + photo)
                image = np.float16(misc.imresize(misc.imread(test_dir + photo), res_sizes[phone])) / 255

                image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
                image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])

                # get enhanced image

                enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
                enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

                before_after = np.hstack((image_crop, enhanced_image))
                photo_name = photo.rsplit(".", 1)[0]

                # save the results as .png images

                misc.imsave("visual_results/" + dirname + "_" + photo_name + "_iteration_" + str(i) + "_enhanced.png",
                            enhanced_image)
                misc.imsave(
                    "visual_results/" + dirname + "_" + photo_name + "_iteration_" + str(i) + "_before_after.png",
                    before_after)
