#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from sklearn.datasets import fetch_mldata
import numpy as np
from AutoEncoder import KooAutoEncoder

# Experiment Parameters
par_hidden = 100    # Hidden Layer Node count
par_visible = 784   # Visible MNIST pix. Image 28 * 28
par_noise = 0.3     # Noise Ratio  0.0 - 0.9
par_untied = False   # Weight Untied (Tied if False)
par_sub_scale = 1000  # if 1000 then only 1/1000 data use in epoch
par_batch_size = 10*2  # it must Multiple of 10. otherwise you bias learning
par_epoch = 2**10   # epoch limit count; each 2**n count up Period(Milestone)
par_dropout = 0.4   # Drop out ratio 0.0 - 0.9
par_train_shuffle = False  # Shuffle the data at the time of training

# ======================================================================================================================
# You can select and execute the actual program from here.
# Since the operation on jupyter notebook, it does not have a main().

# Prepare the data of MNIST.
# ----------------------------------------------------------------------------------------------------------------------
print('---Start---')
print('load MNIST dataset')
mnist = fetch_mldata('MNIST original')

# mnist.data : 70,000 784-dimensional vector data
mnist.data = mnist.data.astype(np.float32)
mnist.data /= 255.0         # 0-1 of the data conversion

# mnist.target : Correct data (teacher data)
mnist.target = mnist.target.astype(np.int32)

N = 60000
y_train, y_test = np.split(mnist.data.copy(), [N])
N_test = y_test.shape[0]

x_train, x_test = np.split(mnist.data, [N])

# ----------------------------------------------------------------------------------------------------------------------
# Data of MNIST has been prepared so far.
# prepare for Expanded learning (ex. Learn Linear mapping. etc)
xtrain_exp = x_train
ytrain_exp = y_train

# Prepare the object of the auto-encoder.
kooae = KooAutoEncoder(n_visible=par_visible,
                       n_hidden=par_hidden,
                       noise=par_noise,
                       untied=par_untied,
                       sub_scale=par_sub_scale,
                       dropout=par_dropout,
                       epoch_limit=par_epoch)
# ----------------------------------------------------------------------------------------------------------------------
train = True
# ----------------------------------------------------------------------------------------------------------------------
if train:
    try:
        # Run the learning of MNIST data
        kooae.train(xtrain_exp, ytrain_exp, epochs=par_epoch, batch_size=par_batch_size)
    except KeyboardInterrupt:
        exit()
        pass

# ----------------------------------------------------------------------------------------------------------------------
# Learning is completed.

# save the Data ----------
if train:
    try:
        kooae.dump_ae("./")
        kooae.reset_report()
    except:
        print ("Unexpected error:")
        raise
# load the Data ----------
if not train:
    try:
        # To report the stored learning results
        print("Loading Object")
        kooae.load_ae("./K160916-2039.kae")  # For example
        # If you restored from the pickle, and re-set the report object.
        kooae.reset_report()
    except:
        print("Unexpected error:")
        raise

# ----------------------------------------------------------------------------------------------------------------------

kooae.set_note("Sample Experiment for Layout explanation")
# Display Weight Number on report (This feature is expected to be extended (select W1 or W2))
w_select = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Report show and save
kooae.show_report(xtrain_exp, ytrain_exp, w_select)

