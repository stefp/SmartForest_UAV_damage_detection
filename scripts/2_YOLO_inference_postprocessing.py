# LOAD LIBRARIES
import torch
import cv2
import os, glob, shutil
import pandas as pd
from ocifs import OCIFileSystem
import fsspec
import PIL.Image as Image
from io import StringIO
from io import BytesIO
from datetime import datetime, timedelta
from osgeo import gdal, osr, ogr
from pathlib import Path
#os.chdir("/home/datascience/myutils")
#from myutils.tools import tile_ortho, yolo2xy, cleanUp_boudingBoxes, mosaic_yoloPred_shp
#from myutils.tools2 import copyFileLocally, copyFileRemotely
import os, glob,shutil
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
from cv2 import cv2
import warnings
warnings.filterwarnings("ignore")      
import subprocess
import sys


import logging
logging.getLogger("ocifs").setLevel(logging.ERROR)
logging.getLogger("yolov5").setLevel(logging.ERROR)
# logging.getLogger("pytorch.core").setLevel(logging.ERROR)

os.system("conda-unpack")
print("Start timestamp in UTC: {}".format(str(datetime.utcnow())))
# import my utils (within the yolov5 environment)

# Switch to activate GPU inference (faster). Of course this requires a VM with GPU.
GPU = False

# My own functions
def copyFileLocally(fileFrom,fileTo):
    fs = OCIFileSystem()
    input = fs.open(fileFrom)
    output = open(fileTo,"wb")
    output.write(input.read())
    output.close()
    input.close()
    
def copyFileRemotely(fileFrom,fileTo):
    fs = OCIFileSystem()
    input = open(fileFrom,"rb")
    output = fs.open(fileTo,"wb")
    output.write(input.read())
    output.close()
    input.close()

# define parameters 
conf_threshold=0.4 # confidence threshold to remove bounding boxes from inference 

intile=1 # tile inner buffer in meters to remove edge artifacts                                       
iou_thresh=0.75 # used to clea-up duplicate boxes. It is the maximum threshold above which two intersecting bounding boxes are considered as the same one and thus the one with largest probability is selected


# define paths to different buckets
# parse input parameters
for item1 in sys.argv[1:]:
    item = item1.split('=',2)
    os.environ[item[0]]=item[1]
    
input_location = os.environ.get("OBJ_INPUT_LOCATION")
output_location = os.environ.get("OBJ_OUTPUT_LOCATION")

if input_location is not None:
    print("Input location: "+ input_location)
else:
    print("Missing input location (OBJ_INPUT_LOCATION)")
    exit
    
if output_location is not None:
    print("Output location: "+ output_location)
else:
    print("Missing ouput location (OBJ_OUTPUT_LOCATION)")
    exit

JOB_RUN_OCID_KEY = "JOB_RUN_OCID"
job_run_ocid = os.environ.get(JOB_RUN_OCID_KEY, "UNDEFINED")

# to work in jobs
dirname = os.path.dirname(os.path.abspath("__file__"))
#model_weights = dirname + '/condapack/yolov5/yolov5/model.pt'
#print(model_weights)

# if executed as a job, the conda pack is in /condapack folder and the artifact in decompressed_artifact
#   decompressed_artifact/nibio/inference.py
#   /home/datascience/condapack/yolov5/lib/python3.8/site-packages/torch/hub.py

# if in Job, the conda pack location is different
if not job_run_ocid or job_run_ocid == "UNDEFINED":
    model_weights = dirname + '/conda/yolov5/yolov5/model.pt'
    yolov5_repo_path="/home/datascience/conda/yolov5/yolov5"
    my_utils_path="/home/datascience/conda/yolov5/utils_stefano" # CHANGE HERE IF CHANGE THE NAME TO THE FOLDER
    from conda.yolov5.utils_stefano.tools import yolo2xy, cleanUp_boudingBoxes, mosaic_yoloPred_shp

    #model = torch.hub.load('/home/datascience/conda/yolov5/yolov5', 'custom', path=model_weights, source='local')
else:
    model_weights = dirname + '/condapack/yolov5/yolov5/model.pt'
    yolov5_repo_path="/home/datascience/condapack/yolov5/yolov5"
    my_utils_path="/home/datascience/condapack/yolov5/utils_stefano" # CHANGE HERE IF CHANGE THE NAME TO THE FOLDER
    from condapack.yolov5.utils_stefano.tools import yolo2xy, cleanUp_boudingBoxes, mosaic_yoloPred_shp

    #model = torch.hub.load('/home/datascience/condapack/yolov5/yolov5', 'custom', path=model_weights, source='local')

print("model :" + model_weights)                        

###########
########### STEP 1. YOLO inference Generate .txt files using Yolov from small .tif files
########### INPUT = .tif files in step_3b_small_tif_files bucket
########### OUTPUT = .txt files in step_3c_temp_text bucket

#for each file in input_location_tif generate .txt files and store in step_3c_temp_text_files
fs = OCIFileSystem()
# notice, this would return no more than 1000 objects in the bucket!
r = fs.glob(input_location + "/*.tif")   
                      
# copy all tif tiles locally for YOLO inference
temp_dir_predict="/home/datascience/temp_predict"
if not os.path.exists(temp_dir_predict):
    os.makedirs(temp_dir_predict)  
    
temp_dir_predict_out="/home/datascience/temp_predict/out"
#if not os.path.exists(temp_dir_predict_out):
#    os.makedirs(temp_dir_predict_out)  
    
temp_dir_labels=temp_dir_predict_out+"/labels"

# copy tiff files from bucket to local
for l in r:
    #print("copy to bucket file : " + temp_dir_predict+ os.path.basename(l))
    copyFileLocally(input_location + "/" + os.path.basename(l), temp_dir_predict+ "/" +os.path.basename(l))

## run inference using the detect.py in the YOLOv5 repo
os.chdir(yolov5_repo_path)
# if GPU switch == True then select the GPU device for inference 
if GPU == True:
    command_predict = "python detect.py --source " + temp_dir_predict + " --weights " + model_weights + " --img " + str(640) + " --name " + temp_dir_predict_out + " --save-txt --save-conf --nosave --conf-thres " + str(conf_threshold) + " --device=0" + " --agnostic"
    #%run /home/datascience/conda/yolov5/yolov5/detect.py --source $temp_dir_predict --weights $model_weights --img 640  --name $temp_dir_predict --save-txt --save-conf --nosave --conf-thres=0.4 --device=0
else:
    command_predict = "python detect.py --source " + temp_dir_predict + " --weights " + model_weights + " --img " + str(640) + " --name " + temp_dir_predict_out + " --save-txt --save-conf --nosave --conf-thres " + str(conf_threshold) + " --agnostic"
    # %run /home/datascience/conda/yolov5/yolov5/detect.py --source $temp_dir_predict --weights $model_weights --img 640  --name $temp_dir_predict --save-txt --save-conf --nosave --conf-thres=0.4

print(os.popen(command_predict).read())


###########
########### Step 2: Parse from image to geographical coordinates and CLEAN-UP bounding boxes
########### INPUT = .tif files in step_3b_small_tif_files bucket + .txt files in step_3c_temp_text bucket
########### OUTPUT = .shp file (1 file) in step_4_geo_shapefiles

# load my own functions
#os.chdir(my_utils_path)
#from utils_stefano.tools import yolo2xy, cleanUp_boudingBoxes, mosaic_yoloPred_shp

gtiffs = glob.glob(temp_dir_predict + "/*.tif")
labels = glob.glob(temp_dir_labels + "/*.txt")

# copy tile index from bucket
#for each file in input_location_tif generate .txt files and store in step_3c_temp_text_files
fs = OCIFileSystem()
# notice, this would return no more than 1000 objects in the bucket!
tile_index_shape = fs.glob(input_location + "/" + "*tile_index*") 
for l in tile_index_shape:
    #print("copying: "+l)
    copyFileLocally(input_location + "/" + os.path.basename(l), temp_dir_predict+ "/" + os.path.basename(l))
    
tile_index_temp = temp_dir_predict+ "/" + os.path.splitext(os.path.basename(l))[0] + ".shp"
print( "path to shapefile="+tile_index_temp+"...........................")
tile_index = gpd.read_file(tile_index_temp) # read tile index shapefile

## Get pixel resolution (in meters) and tile size in pixels
src_ds = gdal.Open(gtiffs[0]) # get raster datasource
_, xres, _, _, _, yres  = src_ds.GetGeoTransform() # get pixel size in meters
tile_size_m=round(src_ds.RasterXSize*xres)
tile_size_px= round(tile_size_m/abs(xres)) # calculate the tile size in pixels
## Get EPSG code
proj = osr.SpatialReference(wkt=src_ds.GetProjection())
EPSG_code= proj.GetAttrValue('AUTHORITY',1)

#dir(gtiffs)

# debug
#print("gtiffs size = " + str(len(gtiffs)))
#print("labels size = " + str(len(labels)))


# iterate through each txt file and trhough each row (bounding box) within a txt file
all_bboxes = None
iter_all=0
for lab in range(len(labels)):
    print(str(round(lab/len(labels)*100))+" % done!")
    # Define one label file and select the corresponding geotiff image
    label_file=labels[lab]
    label_file_name=Path(label_file).stem # ortho name
    for p in gtiffs:
        if Path(p).stem ==label_file_name:
            gtiff_file=p

    #print("debug label file = " + label_file)
    #print("debug tif file = " + gtiff_file)
    
    # determing image witdth and height
    r = gdal.Open(gtiff_file)
    img_width=r.RasterXSize
    img_height=r.RasterYSize
    
    # Convert from yolo coordinates to x1, y1, x2, y2,
    # WARNING : I had to modify yolo2xyx with ocifs + other things
    coords = yolo2xy(label_file, img_width, img_height) # class, x1, y1, x2, y2, probability 

    # Convert from image to geographical coordinates
    ## select tile polygon (from tile index shapefile) that corresponds to the label_file_name
    # the other files are required by the gpd readfile

    
    # tile_index is <class 'geopandas.geodataframe.GeoDataFrame'>
    one_tile=tile_index[tile_index['ID']==label_file_name+".tif"] # Select tile in tile_index that has ID equal to label_file_name

    ## get tile bounding box geographical coordinates (UTM)
    one_tile_XminUTM=one_tile.total_bounds[0]
    one_tile_YminUTM=one_tile.total_bounds[1]
    one_tile_XmaxUTM=one_tile.total_bounds[2]
    one_tile_YmaxUTM=one_tile.total_bounds[3]

    ## take inner buffer equal to the buffer_size_m 
    one_tile_innerB= one_tile
    one_tile_innerB['geometry'] = one_tile_innerB.geometry.buffer(-intile)

    ## get inner tile bounding boxes
    one_tile_inner_XminUTM=one_tile_innerB.total_bounds[0]
    one_tile_inner_YminUTM=one_tile_innerB.total_bounds[1]
    one_tile_inner_XmaxUTM=one_tile_innerB.total_bounds[2]
    one_tile_inner_YmaxUTM=one_tile_innerB.total_bounds[3]

    # Now iterate through each bounding box and assign UTM coordinates and create a shapefile
    bboxes_tile = None
    for i in coords:
        # print("inside coords")
        # Convert bounding box coordinates from image to geographical coords
        X1_UTM=(i[1]*xres)+one_tile_XminUTM
        Y1_UTM=(i[2]*yres)+one_tile_YminUTM+tile_size_m
        X2_UTM=(i[3]*xres)+one_tile_XminUTM
        Y2_UTM=(i[4]*yres)+one_tile_YminUTM+tile_size_m

        # skip bounding box if its centroid is NOT within the inner tile (removing the overlap)
        X_UTM= (X1_UTM+X2_UTM)/2
        Y_UTM= (Y1_UTM+Y2_UTM)/2
        if X_UTM<one_tile_inner_XminUTM or X_UTM>one_tile_inner_XmaxUTM or Y_UTM<one_tile_inner_YminUTM or Y_UTM>one_tile_inner_YmaxUTM:
            #print("continue break")
            continue

        #print("over continue break")    
            
        # Create polygon shape from geographical coords
        lat_point_list = [Y1_UTM, Y1_UTM, Y2_UTM, Y2_UTM, Y1_UTM]
        lon_point_list = [X1_UTM, X2_UTM, X2_UTM, X1_UTM, X1_UTM]
        polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
        crs = {'init': 'epsg:'+EPSG_code}
        data= {'class': [i[0]], 'prob': [i[5]]}
        bbox = gpd.GeoDataFrame(data, crs=crs, geometry=[polygon_geom])
    
        if (bboxes_tile is None):
            bboxes_tile = bbox
        else:
            bboxes_tile = bboxes_tile.append(bbox)
            
    # cleanup boxes (removing overlapping ones)
    if bboxes_tile is not None:
        #clean_boxes = bboxes_tile #cleanUp_boudingBoxes(bboxes_tile, iou_thresh)
        if (all_bboxes is None):
            all_bboxes = bboxes_tile
        else :
            all_bboxes = all_bboxes.append(bboxes_tile)
            
            
# Export predictions as a shapefile and as a geoJSON
#print("Final all_bboxes size = " + type(all_bboxes))
ortho_name = "file"
all_bboxes.to_file(temp_dir_predict + "/" + ortho_name + "_predictions.shp", driver='ESRI Shapefile') # turn this off if it's not needed
#all_bboxes.to_file(temp_dir_predict+"/"+ortho_name+"_predictions.geojson", driver='GeoJSON')

# convert to geoJSON
print("converting geoJSON.............")
input_shp = temp_dir_predict + "/" + ortho_name + "_predictions.shp"
output_geoJson = temp_dir_predict + "/" + ortho_name + "_predictions.json"
cmd = "ogr2ogr -f GeoJSON "  + output_geoJson +" " + input_shp
print(cmd)
subprocess.call(cmd , shell=True)    



###########
########### Step 3: Move shapefile and geoJSON to output bucket
########### INPUT = predictions shapefiles and geojson in local dir
########### OUTPUT = predictions shapefiles and geojson in output bucket



# copy final output shapefile to the output bucket (step_4_geo_shapefiles) # turn off if shapefiles are not needed
predictions_temp_paths=glob.glob(temp_dir_predict + "/*predictions*")
for l in predictions_temp_paths:
    print("copying " + temp_dir_predict + "/" + os.path.basename(l) + " to " + output_location + "/" + os.path.basename(l))
    copyFileRemotely(temp_dir_predict + "/" + os.path.basename(l), output_location + "/" + os.path.basename(l))

# copy final output geoJSON file to the output bucket (step_4_geo_shapefiles)
predictions_json_temp_paths=glob.glob(temp_dir_predict+"/*.json")
copyFileRemotely(temp_dir_predict+"/" +os.path.basename(predictions_json_temp_paths[0]),output_location + "/" + os.path.basename(predictions_json_temp_paths[0]))

# delete temporary folder
shutil.rmtree(temp_dir_predict)

print("End timestamp in UTC: {}".format(str(datetime.utcnow())))
