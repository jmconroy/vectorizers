#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 12:04:02 2020
#from mvlearn.embed.utils import select_dimension

@author: jmconro
"""
import numpy as np
import scipy
from scipy.stats import norm
from sklearn.decomposition import TruncatedSVD
from  mvlearn.embed.utils import select_dimension


def ZG_number_of_topics(doc_term, elbow_index=1, n_topics_upper_bound = 1000):
    """
    Determines an appropriate number of topics to use using 
    Zho and Ghodsi as implemented in graspologic's select_dimension 
    References
    ----------
       [#0] https://graspologic.readthedocs.io/en/latest/reference/embed.html?highlight=select_dimension#graspologic.embed.select_svd
    .. [#1] Zhu, M. and Ghodsi, A. (2006).
        Automatic dimensionality selection from the scree plot via the use of
        profile likelihood. Computational Statistics & Data Analysis, 51(2), 
        pp.918-930.
    
    """
    k_guess = min(n_topics_upper_bound, min(doc_term.shape) - 1)
    if k_guess < 2:
        # there are too few topics here.
        n_topics = 1
        return n_topics
    # SVD
    svd = TruncatedSVD(n_components=k_guess)
    svd.fit(doc_term)

    # Extract singular values
    s = svd.singular_values_[:, None]
    #Use Zho and Ghodsi as implemented in kvlearn's select_dimension
        #turn s into a 1D array and sort it
    s1 = s.ravel()
    s1.sort()
    elbows=select_dimension(s1,n_elbows=max(2,elbow_index+1))
    #Take the  elbow requested
    try:
        n_topics = elbows[0][elbow_index]
    except:
        print('Warning the %dth elbow was not found using dimension %d instead.'%(elbow_index,k_guess))
        n_topics = k_guess  
    return n_topics