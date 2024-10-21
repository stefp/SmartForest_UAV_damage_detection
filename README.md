# Background
This repository includes the methods developed and described in [Puliti and Astrup (2022)](https://www.sciencedirect.com/science/article/pii/S1569843222001431). 

![image](https://user-images.githubusercontent.com/5663984/182541232-ea6de486-c3be-402f-b2a9-1778633e828b.png)

The methods heavily relies on the [YOLOv5 repo](https://github.com/ultralytics/yolov5) by Ultralytics for training the object detector and for inference. 

However, the method was re-adapted to allow working with georeferenced drone RGB orthomosaics. To do so the full pipeline can be subdivided in the following steps:
1) splitting them into overlapping tiles 
2) predicting on each tile
3) converting bounding box coordinates from image to map space
4) aggrregating the predictions for all tiles to produce a wall-to-wall layer (see image below) as a shapefile or GeoJSON

![image](https://user-images.githubusercontent.com/5663984/182541309-fb344a62-8497-4d74-b81e-50c13c146193.png)

# Installation
clone repo
```
git clone https://github.com/SmartForest-no/SmartForest_UAV_damage_detection
cd SmartForest_UAV_damage_detection
```

create new environment
```
conda create -n SmartForest_UAV_damage_detection python=3.10
conda activate SmartForest_UAV_damage_detection

```

Install required packages
- Install gdal (this step might require some fiddling around :) Always fun to install gdal :))
```
conda install -c conda-forge gdal=3.4.0
```

- Other required packages
```
pip install pandas 
pip install fiona
pip install shapely
pip install pyproj
pip install geopandas
pip install geopandas==0.10.2
pip install rasterio==1.2.10

# not sure if there are missing ones from the full requirements
pip install -r requirements.txt
```

In addition:
- Download model weights (*.pt format) at: https://drive.google.com/drive/folders/1SnNRdGXhBFhAyPMKIcxklmw9FTFAp3DI?usp=sharing
- Unzip the two files and place them in the model_zoo folder

# Usage 💻
## Input 🗺️ 
The input is a (high resolution; <= 3-4 cm) georeferenced RGB orthomosaic in GeoTIFF format.

## Output
The output is a multi-polygon vector layer (either shapefile or GeoJSON) covering the entire area (tiles that are on the edges of the orthomosaic are removed from predictions). Each polygon consists of the predicted bounding box for a single tree and is associated with the following fields:

- class: predicted class (healthy, broken, dead)
- prob: predicted probability

## Available models
| model_name  | description | Classes | 
| ------------- | ------------- | ------------- |
| yolov5x_damage_img640_healthy_broken_dead.pt | model used in the original snowbreak detection paper | healthy, broken-top, dead|
| yolov5x_damage_im1280_batch2_healthy_broken_dead_recentDead.pt | retrained model with more bark-beatle affected areas | healthy, broken-top, dead, recently dead|

* suggested model

## How to run 🏃
You can follow the example reported in 'ObjectDetection_damage.ipynb'





