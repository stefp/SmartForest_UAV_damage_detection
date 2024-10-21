# Background
This repository includes the methods developed and described in [Puliti and Astrup (2022)](https://authors.elsevier.com/authorform/landingpage/selection.do?aid=102946&jid=JAG&md5key=d401279a38fa3ae5f3e66d4eeb9dc8fe&lang=English). 

![image](https://user-images.githubusercontent.com/5663984/182541232-ea6de486-c3be-402f-b2a9-1778633e828b.png)

The methods heavily relies on the [YOLOv5 repo](https://github.com/ultralytics/yolov5) by Ultralytics for training the object detector and for inference. 

However, the method was re-adapted to allow working with georeferenced drone RGB orthomosaics. To do so the full pipeline can be subdivided in the following steps:
1) splitting them into overlapping tiles 
2) predicting on each tile
3) converting bounding box coordinates from image to map space
4) aggrregating the predictions for all tiles to produce a wall-to-wall layer (see image below) as a shapefile or GeoJSON

![image](https://user-images.githubusercontent.com/5663984/182541309-fb344a62-8497-4d74-b81e-50c13c146193.png)

# Installation

```
# clone repo
git clone https://github.com/SmartForest-no/snowBreakYOLO
cd snowBreakYOLO

# create new environment
#conda env create -f environment_cnn_wheelRuts.yaml
conda create -n snowBreakYOLO
- Create Conda Environment
odsc conda create -n gdal -s gdal -v 1.0

- Activate the Conda Environment
conda activate /home/datascience/conda/gdal

- List the Conda’s - take a note of the gdal conda location
conda env list

- forge gdal env
conda install -c conda-forge gdal=3.4.0

- Install gdal
pip install pandas 
pip install fiona
pip install shapely
pip install pyproj
pip install geopandas
pip install geopandas==0.10.2
pip install rasterio==1.2.10


# activate the created environment
conda activate snowBreakYOLO

# install requirements
pip install -r requirements.txt
```
In addition:
- Download model weights (*.pt format) at: https://drive.google.com/drive/folders/1SnNRdGXhBFhAyPMKIcxklmw9FTFAp3DI?usp=sharing
- Unzip the two files and and place them in the model folder

# Usage 💻
## Input 🗺️ 
The input consists of a georeferenced drone RGB orthomosaic in GeoTIFF format.

## Output
The output consist of a multi-polygon vector layer (either shapefile or GeoJSON) covering the entire area (tiles that are on the edges of the orthomosaic are removed from predictions). Each polygon consists of the predicted bounding box for a single tree and is associated with the following fields:

- class: predicted class (healthy, broken, dead)
- prob: predicted probability

## Available models
| model_name  | description | Classes |
| ------------- | ------------- | ------------- |
| yolov5x_damage_img640_healthy_broken_dead.pt | model used in the original snowbreak detection paper | healthy, broken-top, dead|
| yolov5x_damage_im1280_batch2_healthy_broken_dead_recentDead.pt* | retrained model with more bark-beatle affected areas | healthy, broken-top, dead, recently dead|

* suggested model

## How to 🏃





