#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:42:39 2019

@author: hwan
"""
import numpy as np
import tensorflow as tf
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def positivity_constraint_exp(x):
    return tf.math.exp(x)

def positivity_constraint_log_exp(x):
    k=0.5
    return (1/k)*tf.math.log(tf.math.exp(k*x)+1)
