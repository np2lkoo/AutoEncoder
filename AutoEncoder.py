#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

# exclude next setp # if you use jupyter notebook
#%matplotlib inline

try:
   import cPickle as pickle
except:
   import pickle

import time
from datetime import datetime
import sys
import copy
import numpy as np
import math
from AEReport import AEReport


# Define AutoEncoder
# ----------------------------------------------------------------------------------------------------------------------
class KooAutoEncoder(object):
    # Constant
    # MINIST Start position at 0 to 9
    k_mnist_start_index = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051, 60000]

    def __init__(self, n_visible=784, n_hidden=784,
                W1=None, W2=None, b1=None, b2=None,
                noise=0.0, untied=False, dropout=0.0, sub_scale=10, epoch_limit=16,
                cal_zero=0.1, cal_mean=0.1, cal_max=0.1):

        # For Record of Intermediate result (each Period(Milestone))
        self.W1rec = []
        self.W2rec = []
        self.b1rec = []
        self.b2rec = []

        self.rng = np.random.RandomState(1)
        self.rng2 = np.random.RandomState(1)
        initRange = np.sqrt(6. / (n_hidden + n_visible + 1))    # 6. is Experience value

        # Initialise of Weight
        if W1 == None:
            self.W1 = self.random_init(initRange, (n_hidden, n_visible))

        if W2 == None:
            if untied:
                W2 = self.random_init(initRange, (n_visible, n_hidden))
            else:
                W2 = self.W1.T

        self.W2 = W2

        # Bias will start from zero
        if b1 == None:
            self.b1 = np.zeros(n_hidden)
        if b2 == None:
            self.b2 = np.zeros(n_visible)

        # Set the activate function
        self.func = Sigmoid() #*
        #self.func = ReLU() #*
        #self.func = Tanh() #*
        #self.func = Liner() #*
        #self.func =StepReLU()  #hybrid
        #self.func =LeakyReLU() #hybrid
        #self.func =LeakyStepReLU() #hybrid

        # Set the match of learning rate in the activation function
        self.alpha = self.func.get_alpha()
        self.alphaBias = self.func.get_alphaBias()
        
        # Set the report object to view the results.
        self.report = AEReport()

        # Remember the parameters.
        self.exp_id = "KYYMMDD-HHMM"   # Dummy
        self.exp_date = "1999/12/31 23:59:59"  # Dummy
        self.version = "1.0"
        self.n_visible = n_visible
        self.n_size = int(math.sqrt(n_visible))
        self.n_hidden = n_hidden
        self.batch_size = 10    # Dummy
        self.noise = noise
        self.untied = untied
        self.epoch_limit = epoch_limit
        self.dropout = dropout
        self.shuffle = 'None'
        self.optimizer = 'None'
        self.beta = 'None'
        self.normalization = 'None'
        self.whitening = 'None'
        self.sub_scale = sub_scale
        self.Wtransport = "init"
        self.option5 = 'None'
        self.option6 = 'None'
        self.option7 = 'None'
        self.option8 = 'None'
        self.costrec = [0]
        self.extime = [0]   # Dummy of init
        self.traintime = 0.0
        self.note = 'None'
        self.cal_zero = cal_zero
        self.cal_mean = cal_mean

        self.m_cal = []

    def random_init(self, r, size):
        return np.array(self.rng.uniform(low=-r, high=r, size=size))

    def corrupt(self, x, noise):
        return self.rng.binomial(size=x.shape, n=1, p=1.0 - noise) * x

    def drop_mask(self, visible, p):
        if p == 0:
            mask = np.ones((visible, self.n_hidden))
        else:
            mask = 1 * np.array(self.rng2.uniform(low=0, high=1, size=(visible, self.n_hidden))) >= p

        adjust = np.sum(mask, axis=1)
        return mask, adjust

    def encode(self, x):
        return self.func.activate_func(np.dot(self.W1, x) + (self.b1))

    def encode_by_snap(self, W, b, x):
        # encode for confirming the intermediate results from later
        return self.func.activate_func(np.dot(W, x) + b)

    def decode(self, y):
        return self.func.activate_func(np.dot(self.W2, y) + (self.b2))

    def decode_by_snap(self, W, b, y):
        # decode for confirming the intermediate results from later
        return self.func.activate_func(np.dot(W, y) + b)

    def get_cost(self, x, z):
        eps = 1e-10
        cost =  - np.sum((x * np.log(z + eps) + (1.-x) * np.log(1.-z + eps))) / self.W1.size  #For Sigmoid
        #cost =  - np.sum((1. / 2) * (x - z) * (z - x)) / self.W1.size    #Square error mean
        return cost

    def get_cost_and_grad(self, x_batch, t_batch, dnum):
        cost = 0.
        grad_W1 = np.zeros(self.W1.shape)
        grad_W2 = np.zeros(self.W2.shape)
        grad_b1 = np.zeros(self.b1.shape)
        grad_b2 = np.zeros(self.b2.shape)

        mask, adjust = self.drop_mask(self.n_visible, self.dropout)
        ii = 0
        for i in range(len(x_batch)):
            x = x_batch[i]
            t = t_batch[i]
            tilde_x = self.corrupt(x, self.noise)
            p = self.encode(tilde_x)
            p = p * mask[ii]
            y = self.decode(p)
            if adjust[ii] != 0:
                y_adj = y * self.n_hidden / adjust[ii]
                y = y_adj * (y_adj < 1.0) + 1.0 * (y_adj >= 1.0)
            else:
                y = t
            cost += self.get_cost(t, y)
            delta1 = - (t - y)

            if self.untied:
                grad_W2 += np.outer(delta1, p)
            else:
                grad_W1 += np.outer(delta1, p).T

            grad_b2 += delta1

            delta2 = np.dot(self.W2.T, delta1) * self.func.differential_func(p) * mask[ii]
            grad_W1 += np.outer(delta2, tilde_x)
            grad_b1 += delta2 * mask[ii]
            ii += 1

        cost /= len(x_batch)
        grad_W1 /= len(x_batch)
        grad_W2 /= len(x_batch)
        grad_b1 /= len(x_batch)
        grad_b2 /= len(x_batch)

        return cost, grad_W1, grad_W2, grad_b1, grad_b2


    def train(self, X, T, epochs=15, batch_size=20):
        print("---Training Start---")
        self.batch_size = batch_size
        self.exp_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.exp_id = datetime.now().strftime("K%y%m%d-%H%M")
        print("====================================")
        print("  Exp-ID   :" + self.exp_id)
        print("  Act f()  :" + self.func.name)
        print("  BatchSize:" + str(self.batch_size))
        print("  Hidden   :" + str(self.n_hidden))
        print("  Noise    :" + str(self.noise))
        print("  DropOut  :" + str(self.dropout))
        print("  EpochLim :" + str(self.epoch_limit))
        print("  SubScale :" + str(self.sub_scale))
        print("  W Trans  :" + self.Wtransport)
        print("====================================")

        # Prepare the data For Calibration
        self.m_cal.append(np.zeros(X[0].shape))
        self.m_cal.append(np.mean(X) * np.ones(X[0].shape))
        main_start_time = time.clock()
        period_start_time = time.clock()

        # Record the initial state
        self.W1rec.append(self.W1.copy())
        self.W2rec.append(self.W2.copy())
        self.b1rec.append(self.b1.copy())
        self.b2rec.append(self.b2.copy())

        # epoch display has from 1, the internal variable from 0.
        batch_num = int(len(X) / (batch_size * self.sub_scale))
        for epoch in range(epochs):
            total_cost = 0.0
            start_time = time.clock()
            for i in range(batch_num):
                ###
                # Loop 1/Subscale
                offset = (epoch * batch_num * (batch_size / 10) + batch_size / 10 * i) % 6000
                subbatch = batch_size / 10
                batch = X[self.k_mnist_start_index[0] + offset:self.k_mnist_start_index[0] + offset + subbatch]
                batchT = T[self.k_mnist_start_index[0] + offset:self.k_mnist_start_index[0] + offset + subbatch]
                # Numbers are to enter the evenly batch.
                for j in range(10)[1:]:
                    batch = np.concatenate([batch, X[self.k_mnist_start_index[j] + offset:self.k_mnist_start_index[j] + offset + subbatch]], axis=0)
                    batchT = np.concatenate([batchT, T[self.k_mnist_start_index[j] + offset:self.k_mnist_start_index[j] + offset + subbatch]], axis=0)

                cost, gradW1, gradW2, gradb1, gradb2 = self.get_cost_and_grad(batch, batchT, len(X))

                total_cost += cost
                self.W1 -= self.alpha * gradW1
                self.W2 -= self.alpha * gradW2
                self.b1 -= self.alpha * self.alphaBias * gradb1
                self.b2 -= self.alpha * self.alphaBias * gradb2

                grad_sum = gradW1.sum() + gradW2.sum() + gradb1.sum() + gradb2.sum()

            end_time = time.clock()
            cost = (1. * self.sub_scale) * total_cost
            print ("epoch = %d: cost = %f: \ttime = %.3f sec" % (epoch + 1, cost, (end_time - start_time)))
            self.costrec.append(cost)
            # Record an intermediate result in the period 2**n - 1
            if epoch in (2 ** np.array(range(20)) - 1):
                print ("/ period = %d ---------- %.3f sec" % ((np.int(np.log2(epoch + 1)) + 1), (end_time - period_start_time)))
                self.W1rec.append(self.W1.copy())
                self.W2rec.append(self.W2.copy())
                self.b1rec.append(self.b1.copy())
                self.b2rec.append(self.b2.copy())
                self.extime.append(end_time - period_start_time)
                period_start_time = end_time

        print ("Total Time = %.3f sec" % (end_time - main_start_time))
        self.traintime = end_time - main_start_time
        print("--- Training has been completed ---")

    def dump_ae(self, save_path):
        with open(save_path + self.exp_id + '.kae', 'wb') as f:
            self.report = None
            d = {
                "AE": self,
            }
            pickle.dump(d, f)

    def load_ae(self, load_file):
        f = open(load_file)
        try:
            load_obj = pickle.load(f)
        except:
            print("Load Error=" + load_file)

        self.aecopy(self, load_obj["AE"])


    def aecopy(self, target, any_obj):
        for attr_name in any_obj.__dict__:
            #print("attr_name=" + attr_name)
            if attr_name == "report":
                print ("bypass report object.")
            else:
                setattr(target, attr_name, getattr(any_obj, attr_name))

    # report the experimental results.
    def show_report(self, my_xtrain, my_ytrain, my_select):
        self.report.show_aereport(self, my_xtrain, my_ytrain, my_select)

    # save the experimental results.
    #def save_report(self):
    #    self.report.save_aereport(self)

    # For the intermediate result acquisition
    def get_W1(self, i):
        return self.W1rec[i]

    def set_W1(self,W1, comment):
        self.W1 = W1
        self.Wtransport = comment

    def get_W2(self, i):
        return self.W2rec[i]

    def set_W2(self, W2):
        self.W2 = W2

    def get_b1(self, i):
        return self.b1rec[i]

    def get_b2(self, i):
        return self.b2rec[i]

    def get_costrec(self):
        return self.costrec

    def reset_report(self):
        self.report = AEReport()

    def set_note(self, str):
        self.note = str

    # MNIST start INDEX acquisition (fixed value)
    def get_mnist_start_index(self, i):
        return self.k_mnist_start_index[i]

# ----------------------------------------------------------------------------------------------------------------------

# This is the original class of the activate function.
# ----------------------------------------------------------------------------------------------------------------------
class ActivateFunction(object):
#    __metaclass__ = ActivateFunctionMeta
    def __init__(self):
        self.name = self.__class__.__name__

    @classmethod
    #@abstractmethod
    def activate_func(cls, x):
        raise NotImplementedError()

    @classmethod
    #@abstractmethod
    def differential_func(cls, x):
        raise NotImplementedError()

    @classmethod
    #@abstractmethod
    def get_alpha(cls):
        raise NotImplementedError()

    @classmethod
    # @abstractmethod
    def get_alphaBias(cls):
        raise NotImplementedError()


# ----------------------------------------------------------------------------------------------------------------------
# Define the activate function
# ----------------------------------------------------------------------------------------------------------------------
class Sigmoid(ActivateFunction):
    def activate_func(cls, x):
        return 1. / (1. + np.exp(-x))

    def differential_func(cls, x):
        return x * (1. - x)

    def get_alpha(cls):
        return 0.26     #Experience value

    def get_alphaBias(cls):
        return 3.10     #Experience value


# ----------------------------------------------------------------------------------------------------------------------
class Tanh(ActivateFunction):
    def activate_func(cls, x):
        return np.tanh(x)

    def differential_func(cls, x):
        return (1. - (np.tanh(x) * np.tanh(x)))

    def get_alpha(cls):
        return 0.04

    def get_alphaBias(cls):
        return 1.00     #Experience value


# ----------------------------------------------------------------------------------------------------------------------
class ReLU(ActivateFunction):
    def activate_func(cls, x):
        return x * (x > 0)

    def differential_func(cls, x):
        return 1. * (x > 0)

    def get_alpha(cls):
        return 0.02

    def get_alphaBias(cls):
        return 1.00     #Experience value


# ----------------------------------------------------------------------------------------------------------------------
class StepReLU(ActivateFunction):
    # Hybrid Type
    def activate_func(cls, x):
        return x * ((x > 0) & (x <= 1)) + 1 * (x > 1)

    def differential_func(cls, x):
        return 1 * ((x > 0) & (x <= 1))

    def get_alpha(cls):
        return 0.022

    def get_alphaBias(cls):
        return 1.00     #Experience value


# ----------------------------------------------------------------------------------------------------------------------
class LeakyReLU(ActivateFunction):
    # Hybrid Type
    leak = 0.2

    def activate_func(cls, x):
        return x * ((x > 0)) + cls.leak * x * (x <= 0)

    def differential_func(cls, x):
        return 1. * ((x > 0)) + cls.leak * ((x <= 0))

    def get_alpha(cls):
        return 0.07

    def get_alphaBias(cls):
        return 1.00     #Experience value


# ----------------------------------------------------------------------------------------------------------------------
class LeakyStepReLU(ActivateFunction):
    # Hybrid Type
    leak = 0.2

    def activate_func(cls, x):
        return x * ((x > 0) & (x <= 1)) + 1 * (x > 1) + cls.leak * x * ((x < 0) | (x > 1)) - cls.leak * (x > 1)

    def differential_func(cls, x):
        return 1 * ((x > 0) & (x <= 1)) + cls.leak * ((x <= 0) | (x > 1))

    def get_alpha(cls):
        return 0.06

    def get_alphaBias(cls):
        return 1.00     #Experience value


# ----------------------------------------------------------------------------------------------------------------------
class Liner(ActivateFunction):
    # Hybrid Type
    def activate_func(cls, x):
        return x

    def differential_func(cls, x):
        return np.ones(x.shape)

    def get_alpha(cls):
        return 0.010

    def get_alphaBias(cls):
        return 1.00     #Experience value


# ----------------------------------------------------------------------------------------------------------------------
class ABS(ActivateFunction):
    # Hybrid Type (This is Joke)
    def activate_func(cls, x):
        return np.abs(x)

    def differential_func(cls, x):
        return np.copysign(np.ones(x.shape), x)

    def get_alpha(cls):
        return 0.01

    def get_alphaBias(cls):
        return 1.00  # Experience value

# ----------------------------------------------------------------------------------------------------------------------
# The definition of the class is over

