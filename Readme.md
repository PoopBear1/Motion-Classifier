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
    |    |   |-- {DOWN SCALE IMAGES} 
    |    |   |
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
* Notice: We use VGG model to find rough key points and their matching, which significantly reduces complexity in finding matches. So please following DFM documents to create the environment -- "https://github.com/ufukefe/DFM".   
* Otherwise: you are free to use any version of python if you want to use traditional SIFT/SURf/ORB only
```shell 
pip install -r requirements.txt
```
* if you want to either run 3x3 matrix version or 3x5 matrix version, please make sure following:
  * line 48 on model.py (kernel size to corresponded version)
  * line 255 on helperfunction.py to corresponded version 

  
#### Brief Description
This Project is designed to do separation of background and foreground via transformation matrices clustering. Specifically, transformation matrices are calculated from multi-frames SIFT features points. By analyzing and clustering on matrices, we can easily determine which part do feature points belong to. The SLIC algorithm is applied to draw a foreground that has at least one feature point.

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
* Joint Learning:
  * Idea 1 
    * 如果学习的是仿射变换矩阵H，
    那么会有一个GT的前景矩阵H可以由GT图提取出来(二值掩图提取)。
    输入是一个前一帧的某坐标(x)以及我们得到的一个GT——H矩阵，通过 公式 x' = H @ x 我们能够得到对应下一帧的坐标(x')
    既然输出是一个第二幅图的估计坐标，损失函数可以用这个预测坐标判断是否在第二张图的ground-truth的范围里面(通过GT images会有一个bbox)
    用简单的Left Min - Right MAX
  * Idea 2
    * 在多尺度下进行联合训练，用scale down的图片同样的方式作匹配得到一系列的运动矩阵，用这系列的矩阵和我们的原尺度图片的系列运动矩阵联合训练
    * 如果因为像素点的减少我们不好找到对应的点所对应的运动矩阵，那么FLIP即是我们的alternative method.
* NMS
  * Idea 1
  * 现在单单是通过前景点的分布密度来判断是否有异常值(假设点会在前景物体上分布更加密集那么一些离散的点应该被排除)。
  * 然而有没有可能性我们使用NMX的方法来更准确的得到这些前景点并且排除一些干扰项？