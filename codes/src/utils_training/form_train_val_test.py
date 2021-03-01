#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:18:47 2019

@author: hwan
"""
import tensorflow as tf

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                 Form Training, Validation and Testing Batches               #
###############################################################################
def form_train_val_test_tf_batches(input_train, output_train,
                                   input_test, output_test,
                                   batch_size, random_seed):
    #=== Casting as float32 ===#
    input_train = tf.cast(input_train,tf.float32)
    output_train = tf.cast(output_train, tf.float32)
    input_test = tf.cast(input_test, tf.float32)
    output_test = tf.cast(output_test, tf.float32)

    #=== Dataset Size ===#
    num_data_train = len(input_train)
    num_data_test = len(input_test)

    #=== Shuffling Data ===#
    input_and_output_train_full = tf.data.Dataset.from_tensor_slices((input_train,
        output_train)).shuffle(num_data_train, seed=random_seed)
    input_and_output_test = tf.data.Dataset.from_tensor_slices((input_test,
        output_test)).batch(batch_size)
    num_batches_test = len(list(input_and_output_test))

    #=== Partitioning Out Validation Set and Constructing Batches ===#
    current_num_data_train = num_data_train
    num_data_train = int(0.8 * num_data_train)
    num_data_val = current_num_data_train - num_data_train
    input_and_output_train = input_and_output_train_full.take(num_data_train).batch(batch_size)
    input_and_output_val = input_and_output_train_full.skip(num_data_train).batch(batch_size)
    num_batches_train = len(list(input_and_output_train))
    num_batches_val = len(list(input_and_output_val))

    return input_and_output_train, input_and_output_val, input_and_output_test,\
           num_batches_train, num_batches_val, num_batches_test

###############################################################################
#                      Form Training and Validation Sets                      #
###############################################################################
def form_train_and_val_set(input_train, output_train):

    #=== Define Input Dimension and Dataset Size ===#
    num_data_train = input_train.shape[0]
    num_data_split = int(0.8 * num_data_train)

    input_val = input_train[num_data_split:]
    output_val = output_train[num_data_split:]
    input_train = input_train[:num_data_split]
    output_train = output_train[:num_data_split]

    return input_train, output_train, input_val, output_val
