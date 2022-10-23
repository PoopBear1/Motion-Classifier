## Contributors:

* Isaac Zhang (u7334258@anu.edu.au)
* Jiawei Li   (u6988392@anu.edu.au)
* Ziwei Cui   (u5643693@anu.edu.au)

## Experimental DataSet
* DAVIS-2016 for moving camera dateset

### Data Representation 
```
${ROOT}
|-- Data
`-- |-- DAVIS
    `-- |-- Annotations
    |    |   |-- bus
    |    |   |   |-- 00000.jpg
    |    |   |   |-- 00001.jpg
    |    |   |   |-- ... 
    |    |   |-- car-roundabout
    |    |   |-- ...
    |    `-- JPEGImages
    |        |-- bus
    |        |   |-- 00000.jpg
    |        |   |-- 00001.jpg
    |        |   |-- ... 
    |        |-- car-roundabout
    |        |-- ...
    |-- Runs
         `-- |-- bus
             |    |-Fundamental_Matrices_npy
             |            `--  Fundamental_on_frame000.npy
             |               |-- ...
             |    |-KP_Matches
             |            `-- 00000_00001_matches.npz
             |               |-- ...
             |-- car-roundabout
             |-- ...
```


### Prerequisites
You need to install the following software/libraries:
* Notice: We use VGG model to find rough key points and their matching, which significantly reduces complexity in Pixel-wise. So please following yolo documents to create the environment -- "https://github.com/ufukefe/DFM".   
* Otherwise: you are free to use any version of python if you want to use traditional SIFT/Surf/ORB only
```shell 
pip install -r requirements.txt
```
* if you want to either run 3x3 matrix version or 3x5 matrix version, please make sure following:
  * line 48 on model.py (kernel size to corresponded version)
  * line 255 on helperfunction.py to corresponded version 

  
#### Brief Description
This Project is designed to do separation of background and foreground via transformation matrices clustering. Specifically, transformation matrices are calculated from multi-frames SIFT features points. By analyzing and clustering on matrices, we can easily determine which part do feature points belong to. The SLIC algorithm is applied to draw a foreground that has at least one SIFT point.

This project has proven that our proposed method not only work on Static Cameras but also actually perform well in Moving Cameras.
    


### To Be fixed:
* Very slow on patch matching if we set feature points threshold super high.(an 80 frames video takes 3 hours to finish collecting all matrices, setting 2k feature points per frame)
* On validation-set, Loss and Accuracy did not smoothly decrease/increase.
* when data amount is small, validation amount might be not enough for efficient evaluation, which causes different result during different training. (Maybe try K-folder Validation latter)


### Update
* add option that input could be either 8 frames (per 10 frame as train set, for an 80 frames video) or 10% of total frame date 7/6/2022  
* change CNN to MLP (replace 2nd,3rd CNN with FNN) 7/7/2022
* change Lewis' SIFT threshold (from 0.65 to 0.4) 7/7/2022
* update ratio Train/Validate from (0.85:0.15) to (0.82 : 0.18)  7/9/2022
* Add constrain that we only form a homography matrix within a pixel and its 100 Neighbor 7/10/2022









## New Idea：
如果学习的是仿射变换，
H矩阵可以由GT+Rasanc提取出来
输入是一个第一幅图坐标+一个H矩阵
输出是一个第二幅图的估计坐标，
损失函数是用这个坐标判断是否在第二张图的ground-truth的范围里面
用简单的Left Min - Right MAX
