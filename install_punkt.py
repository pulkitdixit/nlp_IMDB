# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:29:40 2019

@author: Pulkit Dixit
"""

import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
print('punkt downloaded')