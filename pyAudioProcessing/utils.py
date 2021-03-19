#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:27:32 2021

@author: jsingh
"""
### Imports
import json

### Functions
def write_to_json(file_name, data):
    """
    Write data to file_name.
    """
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile, indent=1)
    print("\nResults saved in {}\n".format(file_name))
