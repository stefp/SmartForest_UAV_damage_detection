import os, glob, shutil
#from ocifs import OCIFileSystem
#import fsspec
from datetime import datetime, timedelta
from osgeo import gdal, ogr, osr
import geopandas as gpd
from pathlib import Path
#import logging
import os
import sys

#logging.getLogger("ocifs").setLevel(logging.ERROR)
#logging.getLogger("yolov5").setLevel(logging.ERROR)
# logging.getLogger("pytorch.core").setLevel(logging.ERROR)
 
#os.system("conda-unpack")

#def copyFileRemotely(fileFrom,fileTo):
#    fs = OCIFileSystem()
#    input = open(fileFrom,"rb")
#    output = fs.open(fileTo,"wb")
#    output.write(input.read())
#    output.close()
#    input.close()
    
#def copyFileLocally(fileFrom,fileTo):
#   fs = OCIFileSystem()
#    input = fs.open(fileFrom)
#    output = open(fileTo,"wb")
#    output.write(input.read())
#    output.close()
#    input.close()
    
#print("Start timestamp in UTC: {}".format(str(datetime.utcnow())))

# parse input parameters
#for item1 in sys.argv[1:]:
#    item = item1.split('=',2)
#    os.environ[item[0]]=item[1]

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

gdal_utils_path="/snowBreakYOLO/scripts/gdal/bin/"

# if executed as a job, the conda pack is in /condapack folder and the artifact in decompressed_artifact
#   decompressed_artifact/nibio/inference.py
#   /home/datascience/condapack/yolov5/lib/python3.8/site-packages/torch/hub.py

#JOB_RUN_OCID_KEY = "JOB_RUN_OCID"
#job_run_ocid = os.environ.get(JOB_RUN_OCID_KEY, "UNDEFINED")

# to work in jobs
dirname = os.path.dirname(os.path.abspath(__file__))
print("dirname= " + dirname)

# DEFINE MAIN PARAMETERS
tile_size_m=32
buffer_size_m=2
format_tiles="GTiff"

## create a temporary directory with a tiles_dir subdirectoruy
local_temp="/home/datascience/tmp/"
if not os.path.exists(local_temp):
    os.makedirs(local_temp)    
    
tiles_dir_temp=local_temp+"tiles_dir/"
if not os.path.exists(tiles_dir_temp):
    os.makedirs(tiles_dir_temp)    
    
#WORKS
fs = OCIFileSystem()
ortho_path = fs.glob(input_location + "*.tif")[0]   ########################################### NEED TO CHANGE THIS TO BATCH_ID.tif
ortho_name=Path(ortho_path).stem # ortho name       ########################################### NEED TO CHANGE THIS TO BATCH_ID

# 1 - COPY ORTHOMOSAIC FROM BUCKET TO LOCAL DIR
# because gdal isn't able to open oci files
input = fs.open(ortho_path)
file_data=input.read()
ortho_path_temp=local_temp + os.path.basename(ortho_path)
## write orthomosaic into the temporary folder
output = open(ortho_path_temp,"w")
with fsspec.open(ortho_path_temp, 'wb') as f:
    f.write(file_data)
    f.close()

# 2 - GET RASTER METADATA
## Get pixel resolution (in meters) and tile size in pixels
src_ds = gdal.Open(ortho_path_temp)
_, xres, _, _, _, yres  = src_ds.GetGeoTransform() # get pixel size in meters
tile_size_px= round(tile_size_m/abs(xres)) # calculate the tile size in pixels
## Get EPSG code
proj = osr.SpatialReference(wkt=src_ds.GetProjection())
EPSG_code= proj.GetAttrValue('AUTHORITY',1)



# run tiling
#tile_ortho(ortho_path,tiles_dir,tile_size_m,buffer_size_m, "GTiff", "/home/datascience/condapack/gdal/bin")


# 3 - GENERATE ORTHOMOSAIC BOUNDARY SHAPEFILE
## Define name for boundary shapefile
shape_path=local_temp+"/"+ortho_name+"_boundary.shp"
## Run gdal_polygonize.py to get boundaries from alpha band (band 4)
#%run /home/datascience/cnn_wheel_ruts/gdal_polygonize.py $ortho_path -b 4 $shape_path
os.chdir(gdal_utils_path)
command_polygonize = "gdal_polygonize.py "+ ortho_path_temp + " -b 4 " + shape_path
print(os.popen(command_polygonize).read())
## Select polygon that has DN equal to 255, indicating the area where drone data is available for
polys = gpd.read_file(shape_path)
polys[polys['DN']==255].to_file(shape_path)

# 4 - TILING THE ORTHOMOSAIC
## Define buffer size and calculate the size of tiles excluding buffer
tile_size_px= round(tile_size_m/abs(xres))
buffer_size_px= round(buffer_size_m/abs(xres))

tileIndex_name=ortho_name+"_tile_index" # define name for output tile index shapefile
## Run gdal_retile.py (can take some minutes) 
os.chdir(gdal_utils_path)
 #%run /home/datascience/cnn_wheel_ruts/gdal_retile.py -targetDir $tiles_dir $ortho_path -overlap $buffer_size_px -ps $tile_size_noBNuffer_px $tile_size_noBNuffer_px -of PNG -co WORLDFILE=YES -tileIndex $tileIndex_name -tileIndexField ID
if format_tiles=="PNG":
    command_retile = "gdal_retile.py -targetDir " + tiles_dir_temp + " " + ortho_path_temp+ " -overlap " + str(buffer_size_px) + " -ps "+str(tile_size_px) + " " + str(tile_size_px) + " -of PNG -co WORLDFILE=YES -tileIndex "+ tileIndex_name + " -tileIndexField ID"
if format_tiles=="GTiff":
    command_retile = "gdal_retile.py -targetDir " + tiles_dir_temp + " " + ortho_path_temp+ " -overlap " + str(buffer_size_px) + " -ps "+str(tile_size_px) + " " + str(tile_size_px) + " -of GTiff -tileIndex "+ tileIndex_name + " -tileIndexField ID"
print(os.popen(command_retile).read())

# 5 - KEEP ONLY TILES WITHIN THE ORTHOMOSAIC BOUNDARY
## Load boundary
boundary = gpd.read_file(shape_path) #  read in the shapefile using geopandas
boundary = boundary.geometry.unary_union #union of all geometries in the GeoSeries
## Load tiles shapefile
tiles = gpd.read_file(tiles_dir_temp+ "/"+ortho_name+"_tile_index.shp")
## Select all tiles that are not within the boundary polygon
tiles_out = tiles[~tiles.geometry.within(boundary)]
## Create a series for each file format with all names of files to be removed
names_tiles_out = [os.path.splitext(x)[0] for x in tiles_out['ID']] # get names without extension
if format_tiles=="PNG":
    pngs_delete=[tiles_dir+ "/"+sub + '.png' for sub in names_tiles_out] # add .png extension
    xml_delete=[tiles_dir+ "/" +sub + '.png.tmp.aux.xml' for sub in names_tiles_out] # ...
    wld_delete=[tiles_dir+ "/"+sub + '.png.wld' for sub in names_tiles_out] #...
    ## Delete files
    for f in pngs_delete: # delete png files
        if os.path.exists(f):
            os.remove(f)
    for f in xml_delete:  # delete xmls files
        if os.path.exists(f):
            os.remove(f)
    for f in wld_delete:  # delete world files
        if os.path.exists(f):
            os.remove(f)
if format_tiles=="GTiff":
    gtiffs_delete=[tiles_dir_temp+ "/"+sub + '.tif' for sub in names_tiles_out] # add .png extension
    for f in gtiffs_delete: # delete png files
        if os.path.exists(f):
            os.remove(f)

# 6 - NOW UPLOAD THE TILES AND THE TILE INDEX SHAPEFILE TO THE THE RESPECTIVE OUTPUT BUCKETS
#import glob
#tiles_to_upload = glob.glob(tiles_dir_temp + "*.")                         
tiles_to_upload = os.listdir(tiles_dir_temp)                       
#tiles_to_upload

for l in tiles_to_upload:
    print("copy to bucket file : " + l)
    copyFileRemotely(tiles_dir_temp+l, output_location + '/' + os.path.basename(l))

# 7 - DELETE TEMPORARY FOLDER
shutil.rmtree(local_temp)

print("End timestamp in UTC: {}".format(str(datetime.utcnow())))
