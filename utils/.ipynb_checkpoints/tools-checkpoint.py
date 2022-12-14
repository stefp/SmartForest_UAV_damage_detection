# Collection of tools/functions useful for object detection and semantic segmentation
import os, glob, shutil
from pathlib import Path
import numpy as np
#from cv2 import cv2
import rasterio as rio
from rasterio.merge import merge
from rasterio.plot import show
from osgeo import gdal, ogr, osr
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

##########################################################################################################################################################################
# function to split a large orthomosaic into small tiles for deep learning inference. The function allows to include a buffer around each tile so that the tiles are actually overlapping
# arguments: 
# - ortho_path    = path pointing at the orthomosaic geotiff file (e.g. "/home/datascience/cnn_wheel_ruts/data/my_ortho.tif")
# - tile_size_m   = length of the side of each tile in meters
# - buffer_size_m = width of the buffer around each tile. This is basically the area of overalp between each tile and is useful to avoid having excessive edge effects when running deep learning inference on small tiles.
# - format_tiles  = export format for tiles (either PNG or GTiff)
##########################################################################################################################################################################
def tile_ortho(ortho_path, tile_size_m, buffer_size_m, format_tiles):
       
    # 1 - DEFINE RASTER AND TILING PARAMETERS
    ## get name of the orthomosaic/drone project and the path where it's stored
    ortho_name=Path(ortho_path).stem # ortho name
    ortho_folder_path=os.path.dirname(ortho_path) # get path name for the folder where the orthomosaic is stored
    ## Get pixel resolution (in meters) and tile size in pixels
    src_ds = gdal.Open(ortho_path) # get raster datasource
    _, xres, _, _, _, yres  = src_ds.GetGeoTransform() # get pixel size in meters
    tile_size_px= round(tile_size_m/abs(xres)) # calculate the tile size in pixels
    ## Get EPSG code
    proj = osr.SpatialReference(wkt=src_ds.GetProjection())
    EPSG_code= proj.GetAttrValue('AUTHORITY',1)
    
    # 2 - GENERATE ORTHOMOSAIC BOUNDARY SHAPEFILE
    ## Define name for boundary shapefile
    shape_path=ortho_folder_path+"/"+ortho_name+"_boundary.shp"
    ## Run gdal_polygonize.py to get boundaries from alpha band (band 4)
    #%run /home/datascience/cnn_wheel_ruts/gdal_polygonize.py $ortho_path -b 4 $shape_path
    os.chdir("/home/datascience/utils/")
    command_polygonize = "gdal_polygonize.py "+ ortho_path + " -b 4 " + shape_path
    print(os.popen(command_polygonize).read())
    ## Select polygon that has DN equal to 255, indicating the area where drone data is available for
    polys = gpd.read_file(shape_path)
    polys[polys['DN']==255].to_file(shape_path)
    
    # 3 - TILING THE ORTHOMOSAIC
    ## Define buffer size and calculate the size of tiles excluding buffer
    buffer_size_m= 2 # size of buffer around each tile 
    tile_size_px= round(tile_size_m/abs(xres))
    buffer_size_px= round(buffer_size_m/abs(xres))
    ## Create folder for tiles to be exported in
    tiles_dir=ortho_folder_path+"/tiles_dir"
    if not os.path.exists(tiles_dir): 
           os.makedirs(tiles_dir)    
    tileIndex_name=ortho_name+"_tile_index" # define name for output tile index shapefile
    ## Run gdal_retile.py (can take some minutes) 
    os.chdir("/home/datascience/utils/")
     #%run /home/datascience/cnn_wheel_ruts/gdal_retile.py -targetDir $tiles_dir $ortho_path -overlap $buffer_size_px -ps $tile_size_noBNuffer_px $tile_size_noBNuffer_px -of PNG -co WORLDFILE=YES -tileIndex $tileIndex_name -tileIndexField ID
    if format_tiles=="PNG":
        command_retile = "gdal_retile.py -targetDir " + tiles_dir + " " + ortho_path+ " -overlap " + str(buffer_size_px) + " -ps "+str(tile_size_px) + " " + str(tile_size_px) + " -of PNG -co WORLDFILE=YES -tileIndex "+ tileIndex_name + " -tileIndexField ID"
    if format_tiles=="GTiff":
        command_retile = "gdal_retile.py -targetDir " + tiles_dir + " " + ortho_path+ " -overlap " + str(buffer_size_px) + " -ps "+str(tile_size_px) + " " + str(tile_size_px) + " -of GTiff -tileIndex "+ tileIndex_name + " -tileIndexField ID"
    print(os.popen(command_retile).read())
    
   
    
    # 4 - KEEP ONLY TILES WITHIN THE ORTHOMOSAIC BOUNDARY
    ## Load boundary
    boundary = gpd.read_file(shape_path) #  read in the shapefile using geopandas
    boundary = boundary.geometry.unary_union #union of all geometries in the GeoSeries
    ## Load tiles shapefile
    tiles = gpd.read_file(tiles_dir+ "/"+ortho_name+"_tile_index.shp")
    ## Select all tiles that are not within the boundary polygon
    tiles_out = tiles[~tiles.geometry.within(boundary)]
    ## Create a series for each file format with all names of files to be removed
    names_tiles_out = [os.path.splitext(x)[0] for x in tiles_out['ID']] # get names without extension
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

##########################################################################################################################################################################
# function to run inference for wheel rut semantic segmentation 
# arguments: 
# - png_dir= directory where png tiles (20 m side) are stored
##########################################################################################################################################################################
def predict_wheelRuts(png_dir):
    from keras_segmentation.predict import predict_multiple #importing predict function from keras 

    # define output path
    output_dir= png_dir+'/predictions'
    
    # create folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # run inference
    predict_multiple( 
      checkpoints_path="/home/datascience/cnn_wheel_ruts/resnet_unet_rgb_20m_grp1_patch" , #path to weights (stored model .json file)
      inp_dir=png_dir , #path to files to be predicted (.png images)
      out_dir=output_dir #path to predicted files - would be in .png format
    )
    
    #import matplotlib.pyplot as plt
    #plt.imshow(out)
    
    
##########################################################################################################################################################################
# function to mosaic raster predictions from semantic segmentation (merging 20 m sized tile prediction)
# arguments: 

# - predicted_dir  = directory where predictions tiles from predict_wheelRuts function (.png format) are stored 
# - dir_orig_tiles = directory where original images from gdal_retile.py  (.png format) are stored 
# - dir_export     = directory where to export the mosaic
##########################################################################################################################################################################
# function to mosaic predictions    
def mosaic_predictions_raster_semantic_seg(
    predicted_dir    # directory where predictions tiles from predict_wheelRuts function (.png format) are stored 
    , dir_orig_tiles # directory where original images from gdal_retile.py  (.png format) are stored 
    , dir_export
    , EPSG_code
    , ortho_name
):   # directory where to export the mosaic
   

    # 1 - PREPARE ENVIRONMENT FOR MOSAIC CREATION
    # move wlds from original images folder to prediction folder
    os.chdir(dir_orig_tiles)
    wlds=[]
    for file in glob.glob("*.wld"):
        predicted_dir_file=dir_orig_tiles+"/"+file
        wlds.append(predicted_dir_file)
    for j in wlds:    
        shutil.move(j, predicted_dir)
    # move xmls from original images folder to prediction folder
    xmls=[]
    for file in glob.glob("*.xml"):
        predicted_dir_file=dir_orig_tiles+"/"+file
        xmls.append(predicted_dir_file)
    for j in xmls:    
        shutil.move(j, predicted_dir)
    # get list of predicted tiles (*.png) 
    os.chdir(predicted_dir)
    pngs=[]
    for file in glob.glob("*.png"):
        pngs.append(file)
    # get list of world files (*.wld) related to the predicted tiles 
    wlds=[]
    for file in glob.glob("*.wld"):
        predicted_dir_file=predicted_dir+"/"+file
        wlds.append(predicted_dir_file)
    # get list of world files (*.wld) related to the predicted tiles 
    xmls=[]
    for file in glob.glob("*.xml"):
        predicted_dir_file=predicted_dir+"/"+file
        xmls.append(predicted_dir_file)
          
    # 2 - CONVERT PNG TILES TO GEOTIFF 
    os.chdir(predicted_dir) # change dir to prediction dir
    # iterate through each png tile and convert it to geotiff using rasterio 
    for i in pngs:
        # get metadata
        ## Load ESRI world file to extract metadata related to the geographical extent the tiles  
        wld_file= f = open(i+'.wld', 'r')
        wld_file=wld_file.read()
        XCellSize =float(wld_file.split()[0])
        YCellSize =float(wld_file.split()[3])
        WorldX=float(wld_file.split()[4])
        WorldY=float(wld_file.split()[5])
        ## Load png image to extract metadata related to the image size
        im = cv2.imread(i)
        Rows=im.shape[0]
        Cols=im.shape[1]
        ## get UTM coords of the upper left and low right corner of the png from the ESRI world file
        XMin = WorldX - (XCellSize / 2)
        YMax = WorldY - (YCellSize / 2) 
        XMax = (WorldX + (Cols * XCellSize)) - (XCellSize / 2)
        YMin = (WorldY + (Rows * YCellSize)) - (YCellSize / 2)

        # conversion
        ## gdal conversion (was working but now it does not trnasfer the coordinates correctly)
        #opt_translate=gdal.TranslateOptions(format="GTiff", bandList=([1]), projWin=[XMin, YMax, XMax,YMin], projWinSRS="EPSG:"+EPSG_code)
        #gdal.Translate(os.path.splitext(i)[0]+".tif", i, options= opt_translate)

        ## rasterio conversion
        dataset = rio.open(i) # open image
        bands = [1] # select only the first band
        data = dataset.read(bands) 
        ### create the output transform 
        west, south, east, north = (XMin, YMin, XMax, YMax)
        transform = rio.transform.from_bounds(west,south,east,north,
                                              data.shape[1],data.shape[2])
        ### set the output image kwargs
        kwargs = {
            "driver": "GTiff",
            "width": data.shape[1], 
            "height": data.shape[2],
            "count": len(bands), 
            "dtype": data.dtype, 
            "nodata": 0,
            "transform": transform, 
            "crs": "EPSG:"+EPSG_code
        }
        with rio.open(os.path.splitext(i)[0]+".tif", "w", **kwargs) as dst:
            dst.write(data, indexes=bands)
            
    # 3 - CREATE MOSAIC (USING EITHER GDAL OR RASTERIO)
    # get list of geotifs
    gtiffs=[]
    for file in glob.glob("*.tif"):
        gtiffs.append(predicted_dir+"/"+file)

    ## mosaic gtiffs using gdal.warp (rasterio seems to be working better!)
    #opt= gdal.WarpOptions(srcNodata=20, multithread=True, resampleAlg="max", srcSRS="EPSG:"+EPSG_code, dstSRS="EPSG:"+EPSG_code)
    #g= gdal.Warp(str(dir_export)+"/"+ortho_name+"mosaic.tif", gtiffs, format="GTiff", options=opt )
    #g = None # Close file and flush to disk
    
    ## mosaic gtiffs using gdal.warp (rasterio seems to be working better!)
    # define output file path and name
    out_fp = str(dir_export)+"/"+ortho_name+"mosaic.tif"
    # List for the source files
    src_files_to_mosaic = []
    # Iterate over raster files and add them to source -list in 'read mode'
    for fp in gtiffs:
        src = rio.open(fp)
        src_files_to_mosaic.append(src)
    # define custom function to merge rasters
    def custom_merge_works(old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None):
        old_data[:] = np.maximum(old_data, new_data)  # <== NOTE old_data[:] updates the old data array *in place*
    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(src_files_to_mosaic, method=custom_merge_works) 
     # Copy the metadata
    out_meta = src.meta.copy()
    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": "EPSG:"+EPSG_code
                     }
                    )
    # Write the mosaic raster to disk
    with rio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic)


    # CLEANUP ENVIRONMENT
    # delete prediction folder
    os.chdir(predicted_dir)
    for j in pngs:    
        os.remove(j)   
    for j in wlds:    
        os.remove(j)
    for j in xmls:    
        os.remove(j) 
    for j in gtiffs:
        os.remove(j)
    shutil.rmtree(predicted_dir) 
    
    # delete dir_orig_tiles
    os.chdir(dir_orig_tiles)
    pngs=[]
    for file in glob.glob("*.png"):
        pngs.append(dir_orig_tiles+"/"+file)    
    for j in pngs:    
        os.remove(j) 
    shutil.rmtree(dir_orig_tiles) 
    
##########################################################################################################################################################################
# YOLO to x1, y1, x2, y2 parsers: converts (x, y, width, height) YOLO format to (x1, y1, x2, y2)  format.
# arguments: 
# - label_file: file with YOLO predictions(s) inside including: class, x, y, width, height, probabilities
# - img_width - width of input image in pixels
# - img_height - height of input image in pixels
#Returns: 
# - a file with a row per predicted bounding box and the following columns: class, x1, y1, x2, y2, probability (note that the coordinates are still in image coordinates and NOT GEOGRAPHICAL ONES)

##########################################################################################################################################################################
    
def yolo2xy(label_file, img_width, img_height):
    """
    Definition: 
Parameters: 
"""
    lfile = open(label_file)
    coords = []
    all_coords = []
    for line in lfile:
        l = line.split(" ")
        label=list(map(float, list(map(float, l[0]))))
        probabs=(l[5])
        #print(probabs)
        coords = list(map(float, list(map(float, l[1:6]))))
        x1 = float(img_width) * (2.0 * float(coords[0]) - float(coords[2])) / 2.0
        y1 = float(img_height) * (2.0 * float(coords[1]) - float(coords[3])) / 2.0
        x2 = float(img_width) * (2.0 * float(coords[0]) + float(coords[2])) / 2.0
        y2 = float(img_height) * (2.0 * float(coords[1]) + float(coords[3])) / 2.0
        tmp = [int(label[0]), int(x1), int(y1), int(x2), int(y2), float(coords[4])]
        all_coords.append(list(tmp))
    lfile.close()
    return all_coords

##########################################################################################################################################################################
# Function to remove duplicate bounding boxes. The function selects couples of bounding boxes that have on a IoU larger than a defined threshold threshold (iou_thresh) and for those selects the one with the largest probability

# arguments: 
# - bboxes_to_clean: polygons representing the bounding boxes as geopandas.geodataframe.GeoDataFrame. The data frame whould cointatin a field named "prob" with predicted probabilities 
# - iou_thresh - threshold above which, two bounding boxes are considered to be a duplicate and thus only the one with largest probability is kept
#Returns: 
# - a new shapefile as geopandas.geodataframe.GeoDataFrame (clean_boxes) where duplicates are removed 

##########################################################################################################################################################################
    

def cleanUp_boudingBoxes(bboxes_to_clean, iou_thresh):
    
    # Assign index
    bboxes_to_clean.index=list(range(len(bboxes_to_clean)))
    # create an emtpy column where to flag the polygons to be removed
    bboxes_to_clean['remove']=0
    for bb in bboxes_to_clean.index:
        #print(bb)
        # select one bounding box
        check_box= bboxes_to_clean.iloc[[bb],:]
        # skip if the row has already been flagged with remove flag
        if any(check_box['remove']==1):
            continue
        # select the rest of the bounding boxes (excluding the selected one)    
        rest=bboxes_to_clean.drop(bboxes_to_clean.index[bb])
        # select bounding boxes that intersect with check_box
        intersect= rest[rest.geometry.map(lambda x: x.intersects(check_box.geometry.any()))]

        # if no boxes intersect then continue to next iteration
        if len(intersect)==0:
            continue
        # otherwise proceed with checking if they have an IoU > defined iou_thresh
        else:    
            iou=[]
            # iterate through each intersecting polygon and check if IoU is larger than a certain threshold 
            for inters in intersect.index:
                # select one of the intersecting polygons
                one_inter= intersect[intersect.index==inters]
                #for this polygon compute the intersection and union with the check_box (target bounding box)
                if(len(one_inter.overlay(check_box, how='intersection'))==0):
                    one_inter['A_inter']=[0]
                    iou.append(float(0))
                else:
                    one_inter['A_inter']=list(one_inter.overlay(check_box, how='intersection').area)
                    one_inter['A_union']=sum(list(one_inter.overlay(check_box, how='union').area))
                    # compute IoU
                    iou.append(float(one_inter['A_inter']/one_inter['A_union']))
            # assign IoU to intersection polygons
            intersect['iou']=iou
            #print(intersect['iou'])

            # if there are some of the intersecting polygons that have an IoU> than the threshold then we need to select either the target polygon (check_box) or the intersecting one
            # the selection is done based on the probability output from the YOLO
            if any(intersect['iou']>iou_thresh):
                candidate_substitute= intersect[intersect['iou']>iou_thresh]
                # if the prob of the candidate is larger than the target polygon then the target poly is removed
                if list(candidate_substitute['prob'])>list(check_box['prob']):
                    bboxes_to_clean.at[bb, 'remove']=1
                # if the prob of the candidate is smaller than the target polygon then the candidate poly is removed
                if list(candidate_substitute['prob'])<list(check_box['prob']):
                    index_candidate= list(candidate_substitute.index)
                    bboxes_to_clean.at[index_candidate[0],'remove']=1

    # keep only boxes with remove==0
    clean_boxes=bboxes_to_clean[bboxes_to_clean['remove']==0]
    
    return clean_boxes



##########################################################################################################################################################################
# Function to parse YOLO predictions to bounding box geographical coordinates (e.g. UTM) and merge them to a single large shapefile
#
# arguments: 
# - tiles_dir: path to where tiled orthomosaic is stored
# - labels_dir: path to where tiled YOLO predictions are stored (.txt)
# - ortho_name: name of the orthomosaic
# - xres: pixels resolution in meters in the x direction
# - yres: pixels resolution in meters in the y direction
# - tile_size_m= length of the side of the tiles in meters
# - intile=1
# - iou_thresh=0.75 
#Returns: 
# - a shapefile as geopandas.geodataframe.GeoDataFrame (all_bboxes) with all YOLO detected bounding boxes and associated class and probability (prob).
##########################################################################################################################################################################
def mosaic_yoloPred_shp(tiles_dir, labels_dir, ortho_name, xres, yres, tile_size_m, EPSG_code, intile=1, iou_thresh=0.75):
    
    # Get list of yolo prediction files (.txt)
    os.chdir(labels_dir)
    labels=[]
    for file in glob.glob("*.txt"):
        labels.append(labels_dir+"/"+file)  
    # Get list of gtiffs (.tif)
    os.chdir(tiles_dir)
    gtiffs=[]
    for file in glob.glob("*.tif"):
        gtiffs.append(tiles_dir+"/"+file)  
    
    # iterate through each prediction file (.txt) and convert YOLO predictions to shapefile
    iter_all=0
    for lab in range(len(labels)):
        print(str(round(lab/len(labels)*100))+" % done!")
        # Define one label file and select the corresponding geotiff image
        label_file=labels[lab]
        label_file_name=Path(label_file).stem # ortho name
        for p in gtiffs:
            if Path(p).stem ==label_file_name:
                gtiff_file=p

        # determing image witdth and height
        r = gdal.Open(gtiff_file)
        img_width=r.RasterXSize
        img_height=r.RasterYSize

        # Convert from yolo coordinates to x1, y1, x2, y2,
        coords= yolo2xy(label_file, img_width, img_height) # class, x1, y1, x2, y2, probability 

        # Convert from image to geographical coordinates
        ## select tile polygon (from tile index shapefile) that corresponds to the label_file_name
        tile_index_path=tiles_dir+"/"+ortho_name+"_tile_index.shp" # define path to tile index
        tile_index=gpd.read_file(tile_index_path) # read tile index shapefile
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
        if iter_all==0:
            iter=0
            for i in coords:
                if iter== 0:
                    # Convert bounding box coordinates from image to geographical coords
                    X1_UTM=(i[1]*xres)+one_tile_XminUTM
                    Y1_UTM=(i[2]*yres)+one_tile_YminUTM+tile_size_m
                    X2_UTM=(i[3]*xres)+one_tile_XminUTM
                    Y2_UTM=(i[4]*yres)+one_tile_YminUTM+tile_size_m

                    # skip bounding box if it's centroid is NOT within the inner tile (removing the overlap)
                    X_UTM= (X1_UTM+X2_UTM)/2
                    Y_UTM= (Y1_UTM+Y2_UTM)/2
                    if X_UTM<one_tile_inner_XminUTM or X_UTM>one_tile_inner_XmaxUTM or Y_UTM<one_tile_inner_YminUTM or Y_UTM>one_tile_inner_YmaxUTM:
                        continue

                    # Create polygon shape from geographical coords
                    lat_point_list = [Y1_UTM, Y1_UTM, Y2_UTM, Y2_UTM, Y1_UTM]
                    lon_point_list = [X1_UTM, X2_UTM, X2_UTM, X1_UTM, X1_UTM]
                    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
                    crs = {'init': 'epsg:'+EPSG_code}
                    data= {'class': [i[0]], 'prob': [i[5]]}
                    bboxes_tile = gpd.GeoDataFrame(data, crs=crs, geometry=[polygon_geom])
                    #bboxes_tile['class']=i[0]
                    #bboxes_tile['prob']=i[5]
                    iter=iter+1

                else :
                    # Convert bounding box coordinates from image to geographical coords
                    X1_UTM=(i[1]*xres)+one_tile_XminUTM
                    Y1_UTM=(i[2]*yres)+one_tile_YminUTM+tile_size_m
                    X2_UTM=(i[3]*xres)+one_tile_XminUTM
                    Y2_UTM=(i[4]*yres)+one_tile_YminUTM+tile_size_m

                    # skip bounding box if it's centroid is NOT within the inner tile (removing the overlap)
                    X_UTM= (X1_UTM+X2_UTM)/2
                    Y_UTM= (Y1_UTM+Y2_UTM)/2
                    if X_UTM<one_tile_inner_XminUTM or X_UTM>one_tile_inner_XmaxUTM or Y_UTM<one_tile_inner_YminUTM or Y_UTM>one_tile_inner_YmaxUTM:
                        continue

                    # Create polygon shape from geographical coords
                    lat_point_list = [Y1_UTM, Y1_UTM, Y2_UTM, Y2_UTM, Y1_UTM]
                    lon_point_list = [X1_UTM, X2_UTM, X2_UTM, X1_UTM, X1_UTM]
                    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
                    crs = {'init': 'epsg:'+EPSG_code}
                    data= {'class': [i[0]], 'prob': [i[5]]}
                    bbox = gpd.GeoDataFrame(data,crs=crs, geometry=[polygon_geom])
                    #bbox['class']=i[0]
                    #bbox['prob']=i[5]
                    # Merge polygons to a single file
                    bboxes_tile = bboxes_tile.append(bbox)
                    iter=iter+1

            # cleanup boxes (removing overlapping ones)
            clean_boxes= cleanUp_boudingBoxes(bboxes_tile, iou_thresh)

            # store boxes in a shapefile with all bounding boxes 
            all_bboxes= clean_boxes
            iter_all=iter_all+1

        else:
            iter=0
            for i in coords:
                if iter== 0:
                    # Convert bounding box coordinates from image to geographical coords
                    X1_UTM=(i[1]*xres)+one_tile_XminUTM
                    Y1_UTM=(i[2]*yres)+one_tile_YminUTM+tile_size_m
                    X2_UTM=(i[3]*xres)+one_tile_XminUTM
                    Y2_UTM=(i[4]*yres)+one_tile_YminUTM+tile_size_m

                    # skip bounding box if it's centroid is NOT within the inner tile (removing the overlap)
                    X_UTM= (X1_UTM+X2_UTM)/2
                    Y_UTM= (Y1_UTM+Y2_UTM)/2
                    if X_UTM<one_tile_inner_XminUTM or X_UTM>one_tile_inner_XmaxUTM or Y_UTM<one_tile_inner_YminUTM or Y_UTM>one_tile_inner_YmaxUTM:
                        continue

                    # Create polygon shape from geographical coords
                    lat_point_list = [Y1_UTM, Y1_UTM, Y2_UTM, Y2_UTM, Y1_UTM]
                    lon_point_list = [X1_UTM, X2_UTM, X2_UTM, X1_UTM, X1_UTM]
                    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
                    crs = {'init': 'epsg:'+EPSG_code}
                    data= {'class': [i[0]], 'prob': [i[5]]}
                    bboxes_tile = gpd.GeoDataFrame(data, crs=crs, geometry=[polygon_geom])
                    #bboxes_tile['class']=i[0]
                    #bboxes_tile['prob']=i[5]
                    iter=iter+1
                else :
                    # Convert bounding box coordinates from image to geographical coords
                    X1_UTM=(i[1]*xres)+one_tile_XminUTM
                    Y1_UTM=(i[2]*yres)+one_tile_YminUTM+tile_size_m
                    X2_UTM=(i[3]*xres)+one_tile_XminUTM
                    Y2_UTM=(i[4]*yres)+one_tile_YminUTM+tile_size_m

                    # skip bounding box if it's centroid is NOT within the inner tile (removing the overlap)
                    X_UTM= (X1_UTM+X2_UTM)/2
                    Y_UTM= (Y1_UTM+Y2_UTM)/2
                    if X_UTM<one_tile_inner_XminUTM or X_UTM>one_tile_inner_XmaxUTM or Y_UTM<one_tile_inner_YminUTM or Y_UTM>one_tile_inner_YmaxUTM:
                        continue

                    # Create polygon shape from geographical coords
                    lat_point_list = [Y1_UTM, Y1_UTM, Y2_UTM, Y2_UTM, Y1_UTM]
                    lon_point_list = [X1_UTM, X2_UTM, X2_UTM, X1_UTM, X1_UTM]
                    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
                    crs = {'init': 'epsg:'+EPSG_code}
                    data= {'class': [i[0]], 'prob': [i[5]]}
                    bbox = gpd.GeoDataFrame(data,crs=crs, geometry=[polygon_geom])
                    #bbox['class']=i[0]
                    #bbox['prob']=i[5]
                    # Merge polygons to a single file
                    bboxes_tile = bboxes_tile.append(bbox)
                    iter=iter+1

            # cleanup boxes (removing overlapping ones)
            clean_boxes=cleanUp_boudingBoxes(bboxes_tile, iou_thresh)

            # store boxes in a shapefile with all bounding boxes 
            all_bboxes = all_bboxes.append(clean_boxes)
            iter_all=iter_all+1
            
            return all_bboxes
