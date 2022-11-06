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
    |    |   |-- bus -- {DOWN SCALE IMAGES} 
    |    |   |    `--- 1080p
    |    |   |        |-- 00000.jpg
    |    |   |        |-- 00001.jpg
    |    |   |        |-- ...
    |    |   |    
    |    |   |-- car-roundabout
    |    |   |-- ...
    |    `-- JPEGImages
    |        |-- bus  -- {DOWN SCALE IMAGES} 
    |        |   `--- 1080p 
    |     `  |          |-- 00000.jpg 
    |        |          |-- 00001.jpg
    |        |          |-- ... 
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

### Test our code
you need to follow above Data Representation. Since we have discard using mult-scale image training, you may simply put testing video on "../Data/JPEGImages/${Video Name}/1080p" and then out its labels in "../Data/Annotations/${Video Name}/1080p"
If you first time run our code(means you don't have any data,please make sure -generate_data is set on "True" and  -reuse is on "False")
e.g.:
```shell
python new_main.py --generate_data True --matrix_type Homography --reuse False
```
otherwise you can use following commands 
```shell
python new_main.py --generate_data False --matrix_type Homography --reuse True
```

#### Brief Description
This Project is designed to do separation of background and foreground via transformation matrices clustering. Specifically, transformation matrices are calculated from multi-frames SIFT features points. By analyzing and clustering on matrices, we can easily determine which part do feature points belong to. The SLIC algorithm is applied to draw a foreground that has at least one feature point.

This project has proven that our proposed method not only work on Static Cameras but also actually perform well in Moving Cameras.
    


### To Be fixed:
* Very slow on patch matching if we set feature points threshold super high.(an 80 frames video takes 3 hours to finish collecting all matrices, setting 2k feature points per frame)
  * fixed with importing dfm feature matching
* On validation-set, Loss and Accuracy did not smoothly decrease/increase.
* when data amount is small, validation amount might be not enough for efficient evaluation, which causes different result during different training. 
  * (Maybe try K-folder Validation latter)
  * fixed with importing dfm feature matching
  



