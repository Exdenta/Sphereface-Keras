# SphereFace: Deep Hypersphere Embedding for Face Recognition

A Keras Implementation of SphereFace.

[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)

Pretrained models: 
- Original Sphereface-20 Caffe model from [wy1iu/sphereface](https://github.com/wy1iu/sphereface): [Google Drive](https://drive.google.com/file/d/0B_geeR2lTMegb2F6dmlmOXhWaVk/view) link

## Prerequisites:
- Python 3

## Train:
Download the training set **CASIA-WebFace**, unzip it and put it to datasets/.<br>
[Google Drive](https://drive.google.com/open?id=1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz) link from [here](https://github.com/happynear/AMSoftmax/issues/18)
```Shell
./datasets/get_casia.sh
# Preprocess:
python3 preprocess/detect_and_align.py --source_dir datasets/CASIA-WebFace --save_dir datasets/CASIA-WebFace-112x96
python3 train/form_train_data.py
jupyter notebook
# go to train/train.ipynb
```

## Test:
Download the training set **LFW** and put it to datasets/
```Shell
./datasets/get_lfw.sh
# Preprocess:
python3 preprocess/detect_and_align.py --source_dir datasets/lfw --save_dir datasets/lfw_112x96
```

Run test:
```Shell
python3 test/evaluate.py
```
