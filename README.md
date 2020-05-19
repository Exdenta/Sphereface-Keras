# SphereFace: Deep Hypersphere Embedding for Face Recognition

A Keras Implementation of SphereFace.

[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)

Pretrained models: 
- Original Sphereface-20 Caffe model from [wy1iu/sphereface](https://github.com/wy1iu/sphereface): [Google Drive](https://drive.google.com/file/d/0B_geeR2lTMegb2F6dmlmOXhWaVk/view) link

## Prerequisites:
- Python 3

## Train:
Download the training set **CASIA-WebFace**, unzip it and put it to train/data/.<br>
[Google Drive](https://drive.google.com/open?id=1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz) link from [here](https://github.com/happynear/AMSoftmax/issues/18)

Preprocess:
```Shell
python preprocess/detect_and_align.py --source_dir train/data/CASIA-WebFace --save_dir train/data/CASIA-WebFace-112x96
```

## Test:
Download the training set **LFW** and put it to test/data/
```Shell
./preprocess/get_lfw.sh
tar xvf data/lfw.tgz -C data/
./test/get_pairs.sh
```

Preprocess:
```Shell
python preprocess/detect_and_align.py --source_dir test/data/lfw --save_dir test/data/lfw_112x96
```

Run test:
```Shell
python test/evaluate.py
```
