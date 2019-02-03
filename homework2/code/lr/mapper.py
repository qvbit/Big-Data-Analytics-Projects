#!/usr/bin/env python

import sys
import random

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-n", "--model-num", action="store", dest="n_model",
                  help="number of models to train", type="int")
parser.add_option("-r", "--sample-ratio", action="store", dest="ratio",
                  help="ratio to sample for each ensemble", type="float")

options, args = parser.parse_args(sys.argv)

random.seed(6505)

for line in sys.stdin:
    # TODO
    # Note: The following lines are only there to help 
    #       you get started (and to have a 'runnable' program). 
    #       You may need to change some or all of the lines below.
    #       Follow the pseudocode given in the PDF.
    M = options.n_model
    for i in range(1, M+1):
        m = random.random() # number between 0 and 1
        if m < options.ratio:
            value = line.strip() # Note that value here is one training example.
            if value:
                print("%d\t%s" % (i, value))