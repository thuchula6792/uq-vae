#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def value_to_string(value):
    if value >= 1:
        value = int(value)
        string = str(value)
    elif value == 0:
        string = '0'
    else:
        value_decimal_form = '%.9f'%(value)
        string = 'pt'

        for n in range(1,len(value_decimal_form)):
            if value_decimal_form[-n] != '0':
                index = n
                break

        string += value_decimal_form[2:12-index]

    return string
