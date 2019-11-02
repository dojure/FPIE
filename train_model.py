# python train_model.py model={iphone,sony,blackberry} dped_dir=dped vgg_dir=vgg_pretrained/imagenet-vgg-verydeep-19.mat
import os
import sys

import numpy as np
import tensorflow as tf
from scipy import misc

import models
import utils
import vgg
from load_dataset import load_test_data, load_batch
from ssim import MultiScaleSSIM

# defining size of the training image patches

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

# processing command arguments

phone, batch_size, train_size, learning_rate, num_train_iters, \
w_content, w_color, w_texture, w_tv, \
dped_dir, vgg_dir, eval_step, run, kernel_size, depth, blocks, \
parametric, s_conv, convdeconv = utils.process_command_args(sys.argv)

dirname = phone + "_" + run
if run == "":
    dirname = phone

np.random.seed(0)

# loading training and test data

print("Loading testing data...")
test_data, test_answ = load_test_data(phone, dped_dir, PATCH_SIZE)

print("Loading training data...")
train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)

TEST_SIZE = test_data.shape[0]
num_test_batches = int(test_data.shape[0] / batch_size)

# defining system architecture
print("Initializing...")
with tf.Graph().as_default(), tf.Session() as sess:
    # placeholders for training data
    print("Session initialized")
    phone_ = tf.placeholder(tf.float32, [None, PATCH_SIZE], name="train-phone")
    phone_image = tf.reshape(phone_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3], name="train-phone-img")

    dslr_ = tf.placeholder(tf.float32, [None, PATCH_SIZE], name="train-dslr")
    dslr_image = tf.reshape(dslr_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3], name="train-dslr-img")

    adv_ = tf.placeholder(tf.float32, [None, 1], name="cointoss")

    # get processed enhanced image

    if convdeconv:
        enhanced = models.convdeconv(phone_image, depth, parametric=parametric, s_conv=s_conv)
    else:
        enhanced = models.resnet(phone_image, kernel_size, depth, blocks, parametric=parametric, s_conv=s_conv)

    # 2) content loss

    with tf.name_scope("content_loss"):
        CONTENT_LAYER = 'relu5_4'

        enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
        dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_image * 255))

        content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
        loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size

        tf.summary.scalar("loss_content", loss_content)

    # transform both dslr and enhanced images to grayscale

    enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, PATCH_WIDTH * PATCH_HEIGHT])
    dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(dslr_image), [-1, PATCH_WIDTH * PATCH_HEIGHT])

    # push randomly the enhanced or dslr image to an adversarial CNN-discriminator

    adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)
    adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])

    discrim_predictions = models.adversarial(adversarial_image)

    # losses

    # 1) texture (adversarial) loss

    with tf.name_scope("texture_loss"):
        discrim_target = tf.concat([adv_, 1 - adv_], 1)

        loss_discrim = -tf.reduce_sum(discrim_target * tf.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))
        loss_texture = -loss_discrim

        correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
        discrim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        tf.summary.scalar("loss_texture", loss_texture)
        tf.summary.scalar("discrim_accuracy", discrim_accuracy)

    # 3) color loss

    with tf.name_scope("color_loss"):
        enhanced_blur = utils.blur(enhanced)
        dslr_blur = utils.blur(dslr_image)

        loss_color = tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2)) / (2 * batch_size)

        tf.summary.scalar("loss_color", loss_color)

    # 4) total variation loss

    with tf.name_scope("tv_loss"):
        batch_shape = (batch_size, PATCH_WIDTH, PATCH_HEIGHT, 3)
        tv_y_size = utils._tensor_size(enhanced[:, 1:, :, :])
        tv_x_size = utils._tensor_size(enhanced[:, :, 1:, :])
        y_tv = tf.nn.l2_loss(enhanced[:, 1:, :, :] - enhanced[:, :batch_shape[1] - 1, :, :])
        x_tv = tf.nn.l2_loss(enhanced[:, :, 1:, :] - enhanced[:, :, :batch_shape[2] - 1, :])
        loss_tv = 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

        tf.summary.scalar("loss_tv", loss_tv)

    # final loss

    with tf.name_scope("final_loss"):
        loss_generator = w_content * loss_content + w_texture * loss_texture + w_color * loss_color + w_tv * loss_tv

        tf.summary.scalar("loss_generator", loss_generator)

    # psnr loss
    with tf.name_scope("psnr_loss"):
        enhanced_flat = tf.reshape(enhanced, [-1, PATCH_SIZE])

        loss_mse = tf.reduce_sum(tf.pow(dslr_ - enhanced_flat, 2)) / (PATCH_SIZE * batch_size)
        loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))

        tf.summary.scalar("loss_psnr", loss_psnr)

    # optimize parameters of image enhancement (generator) and discriminator networks

    with tf.name_scope("train"):
        generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
        discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]

        train_step_gen = tf.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)
        train_step_disc = tf.train.AdamOptimizer(learning_rate).minimize(loss_discrim, var_list=discriminator_vars)

    saver = tf.train.Saver(var_list=generator_vars, max_to_keep=500)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./summaries/" + dirname, sess.graph)
    test_writer = tf.summary.FileWriter("./summaries/" + dirname + "_test")

    print('Initializing variables')
    sess.run(tf.global_variables_initializer())

    # resume from saved model
    if True:
        # states = tf.train.get_checkpoint_state("models")
        # checkpoint_paths = states.all_model_checkpoint_paths
        # print(checkpoint_paths)
        # import re
        # regex = re.compile(r'\d+')
        # iters = [int(regex.findall(filename)[-1]) for filename in checkpoint_paths]
        # iteration = max(iters)
        # print(iteration)
        # saver.recover_last_checkpoints(checkpoint_paths)
        if not os.path.exists("models"):
            os.makedirs("models")
        files = os.listdir("models")
        import re

        regex = re.compile(r'\d+')
        lis = [(filename, int(regex.findall(filename)[-1])) for filename in files if
               dirname + "_" in filename and ".ckpt.index" in filename]
        import operator

        if len(lis) > 0:
            filename, iteration = max(lis, key=operator.itemgetter(1))
            saver.restore(sess, "models/" + filename[:-6])
            print("restored", filename[:-6], iteration)
        else:
            iteration = 0

    train_loss_gen = 0.0
    train_acc_discrim = 0.0

    all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])
    test_crops = test_data[np.random.randint(0, TEST_SIZE, 5), :]

    logs = open('models/' + dirname + '.txt', "w+")
    logs.close()

    gen_parameters = 0
    disc_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        if variable.name.startswith("generator"):
            gen_parameters += variable_parameters
        if variable.name.startswith("discriminator"):
            disc_parameters += variable_parameters
    print("Trainable parameters: gen:", gen_parameters, "disc:", disc_parameters)
    np.random.seed(num_train_iters)
    print('Training network')

    for i in range(iteration + 1, num_train_iters + 1):
        print("step", i)
        # train generator

        idx_train = np.random.randint(0, train_size, batch_size)

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        [summary, loss_temp, _] = sess.run([merged, loss_generator, train_step_gen],
                                           feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: all_zeros})
        train_loss_gen += loss_temp / eval_step

        writer.add_summary(summary, i)

        # train discriminator

        idx_train = np.random.randint(0, train_size, batch_size)

        # generate image swaps (dslr or enhanced) for discriminator
        swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        [summary, accuracy_temp, _] = sess.run([merged, discrim_accuracy, train_step_disc],
                                               feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})
        train_acc_discrim += accuracy_temp / eval_step

        writer.add_summary(summary, i)

        if i % eval_step < 1 and i != 0:
            # print("test step")

            # test generator and discriminator CNNs

            test_losses_gen = np.zeros((1, 6))
            test_accuracy_disc = 0.0
            loss_ssim = 0.0

            for j in range(num_test_batches):
                # print('test', j, 'out of', num_test_batches)
                be = j * batch_size
                en = (j + 1) * batch_size

                swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

                phone_images = test_data[be:en]
                dslr_images = test_answ[be:en]

                [enhanced_crops, accuracy_disc, losses] = sess.run([enhanced, discrim_accuracy,
                                                                    [loss_generator, loss_content, loss_color,
                                                                     loss_texture, loss_tv, loss_psnr]],
                                                                   feed_dict={phone_: phone_images,
                                                                              dslr_: dslr_images, adv_: swaps})

                test_losses_gen += np.asarray(losses) / num_test_batches
                test_accuracy_disc += accuracy_disc / num_test_batches

                loss_ssim += MultiScaleSSIM(np.reshape(dslr_images * 255, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 3]),
                                            enhanced_crops * 255) / num_test_batches

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="texture_loss/discrim_accuracy", simple_value=test_accuracy_disc),
                tf.Summary.Value(tag="final_loss/loss_generator", simple_value=test_losses_gen[0][0]),
                tf.Summary.Value(tag="content_loss/loss_content", simple_value=test_losses_gen[0][1]),
                tf.Summary.Value(tag="color_loss/loss_color", simple_value=test_losses_gen[0][2]),
                tf.Summary.Value(tag="texture_loss/loss_texture", simple_value=test_losses_gen[0][3]),
                tf.Summary.Value(tag="tv_loss/loss_tv", simple_value=test_losses_gen[0][4]),
                tf.Summary.Value(tag="psnr_loss/loss_psnr", simple_value=test_losses_gen[0][5]),
                tf.Summary.Value(tag="ssim", simple_value=loss_ssim),
            ])
            test_writer.add_summary(summary, i)

            logs_disc = "step %d, %s | discriminator accuracy | train: %.4g, test: %.4g" % \
                        (i, phone, train_acc_discrim, test_accuracy_disc)

            logs_gen = "generator losses | train: %.4g, test: %.4g | content: %.4g, color: %.4g, texture: %.4g, tv: %.4g | psnr: %.4g, ssim: %.4g\n" % \
                       (train_loss_gen, test_losses_gen[0][0], test_losses_gen[0][1], test_losses_gen[0][2],
                        test_losses_gen[0][3], test_losses_gen[0][4], test_losses_gen[0][5], loss_ssim)

            print(logs_disc)
            print(logs_gen)

            # save the results to log file

            logs = open('models/' + dirname + '.txt', "a")
            logs.write(logs_disc)
            logs.write('\n')
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # save visual results for several test image crops

            enhanced_crops = sess.run(enhanced, feed_dict={phone_: test_crops, dslr_: dslr_images, adv_: all_zeros})

            if not os.path.exists("results"):
                os.makedirs("results")
            idx = 0
            for crop in enhanced_crops:
                before_after = np.hstack((np.reshape(test_crops[idx], [PATCH_HEIGHT, PATCH_WIDTH, 3]), crop))
                misc.imsave('results/' + str(dirname) + "_" + str(idx) + '_iteration_' + str(i) + '.jpg', before_after)
                idx += 1

            train_loss_gen = 0.0
            train_acc_discrim = 0.0

            # save the model that corresponds to the current iteration

            if not os.path.exists("models"):
                os.makedirs("models")
            saver.save(sess, 'models/' + str(dirname) + '_iteration_' + str(i) + '.ckpt', write_meta_graph=False)

            # reload a different batch of training data

            del train_data
            del train_answ
            train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
