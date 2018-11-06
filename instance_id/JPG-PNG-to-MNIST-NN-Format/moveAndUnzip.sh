#!/bin/bash
gzip -d test-images-idx3-ubyte.gz
gzip -d test-labels-idx1-ubyte.gz
gzip -d train-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz

mv test-images-idx3-ubyte ../tensorflow-triplet-loss/test-images-idx3-ubyte
mv test-labels-idx1-ubyte ../tensorflow-triplet-loss/test-labels-idx1-ubyte
mv train-images-idx3-ubyte ../tensorflow-triplet-loss/train-images-idx3-ubyte
mv train-labels-idx1-ubyte ../tensorflow-triplet-loss/train-labels-idx1-ubyte