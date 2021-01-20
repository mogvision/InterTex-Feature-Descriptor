# InterTex: Interwoven Texture-Based Feature Descriptor

## Introduction
InterTex is a fast feature descriptor for computer vision tasks. This repo includes the code for feature extraction from the keypoint detected by FFD (https://github.com/mogvision/FFD). Given a pair of images, you can use this repo to extract matching features across the image pair.

*  Link: https://www.sciencedirect.com/science/article/pii/S003132032100008X?via%3Dihub#

*  PDF: https://1drv.ms/b/s!Ap1FyV7QV37ThU92tMnxL5NEaRyq?e=mRoQLN

* Authors: Morteza Ghahremani, Yitian Zhao, Bernard Tiddeman and Yonghuai Liu



## Dependencies
* Python 3 >= 3.5
* OpenCV >= 3.4 
* NumPy >= 1.18


## Contents
There are two main scripts in this repo:

1. `demo.py`: runs and shows extracted keypoints located in `image/`
2. `match_pairs.py`: reads an image pair from `image/` and matches (FFD is used for feature detection)

```sh
python3 demo.py
python3 match_pairs.py
```

P.S. If you get error: "./InterTexFFD: Permission denied", please just run `chmod 777 InterTexFFD'.

## BibTeX Citation
If you use any ideas from the paper or code from this repo, please consider citing:

```txt
@article{GHAHREMANI2021107821,
title = "Interwoven Texture-Based Description of Interest Points in Images",
author = "Morteza Ghahremani and Yitian Zhao and Bernard Tiddeman and Yonghuai Liu",
journal = "Pattern Recognition",
pages = "107821",
year = "2021",
issn = "0031-3203",
doi = "https://doi.org/10.1016/j.patcog.2021.107821",
url = "http://www.sciencedirect.com/science/article/pii/S003132032100008X",
}
```
