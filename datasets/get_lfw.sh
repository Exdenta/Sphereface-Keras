#!/bin/bash

wget -O datasets/lfw.tgz http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar xvf datasets/lfw.tgz -C datasets/
wget -O datasets/lfw_pairs.txt http://vis-www.cs.umass.edu/lfw/pairs.txt
