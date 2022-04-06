"""

This code is created by Ridvan Salih KUZU @DLR
 and modified by Mohammad Alasawedah  @MEOTEQ team
LAST EDITED:  14.12.2021
ABOUT SCRIPT:

"""

import os
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features
from tqdm import tqdm
import pickle
import math
from rasterio.windows import Window
import glob
import copy


def S1Extractor(rootpath, label_dir, npyfolder, data_type='train'):
  """
  :param rootpath: directory of input images
  :param label_dir: directory of ground-truth polygons in GeoJSON format
  :param npyfolder: folder to save the field data for each field polygon
  :param data_type: type of data train/test
  
  """
  with open(os.path.join(rootpath, "bbox.pkl"), 'rb') as f:
    bbox = pickle.load(f)
    crs = str(bbox.crs)
    minx, miny, maxx, maxy = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
  
  labels = gpd.read_file(label_dir)
  labels = labels.to_crs(crs)
    
  vv = np.load(os.path.join(rootpath, "vv.npy"))
  vh = np.load(os.path.join(rootpath, "vh.npy"))
  bands = np.stack([vv[:,:,:,0],vh[:,:,:,0]], axis=3)
  _, width, height, _ = bands.shape
  transform = rio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
  bands=bands.transpose(0, 3, 1, 2)

  fid_mask = features.rasterize(zip(labels.geometry, labels.fid), all_touched=True,
                            transform=transform, out_shape=(width, height))
  assert len(np.unique(fid_mask)) > 0, f"WARNING: Vectorized fid mask contains no fields. " \
                                        f"Does the label geojson {label_dir.split('/')[-1]} cover the region defined by {rootpath}?"

  crop_mask = features.rasterize(zip(labels.geometry, labels.crop_id), all_touched=True,
                            transform=transform, out_shape=(width, height))
  assert len(np.unique(crop_mask)) > 0, f"WARNING: Vectorized fid mask contains no fields. " \
                                        f"Does the label geojson {label_dir.split('/')[-1]} cover the region defined by {rootpath}?"

  for index, feature in tqdm(labels.iterrows(), total=len(labels), position=0, leave=True, desc="INFO: Extracting Sentinel-1 time series"):
    label = feature.crop_id
    fid = feature.fid
    #coords = feature.geometry.centroid.coords
    npyfile = f'{npyfolder}/{data_type}/s1/{label-1}/{fid}.npz'
    if data_type == 'test':
      npyfile = f'{npyfolder}/{data_type}/s1/{fid}.npz'
    if not os.path.exists(npyfile):
      left, bottom, right, top = feature.geometry.bounds
      window = rio.windows.from_bounds(left, bottom, right, top, transform)
      row_off = window.row_off
      col_off = window.col_off
      height = window.height
      width = window.width
      if row_off <0:
        #height = height + row_off
        row_off = 0
      if col_off <0:
        col_off = 0 
      row_start = math.floor(row_off)
      row_end = math.floor(row_off) + math.ceil(height)
      col_start = math.floor(col_off)
      col_end = math.floor(col_off) + math.ceil(width)
      image_stack = bands[:, :,row_start:row_end, col_start:col_end]
      #image_stack = bands[:, :, col_start:col_end, row_start:row_end]
      
      mask = copy.deepcopy(fid_mask[row_start:row_end, col_start:col_end])
        #mask = fid_mask[col_start:col_end, row_start:row_end]
        
      mask[mask != feature.fid] = 0
      mask[mask == feature.fid] = 1
      os.makedirs(npyfolder, exist_ok=True)
      np.savez(npyfile, image_stack=image_stack.astype(np.float32), mask=mask.astype(np.float32), feature=feature.drop("geometry").to_dict())  


def S2Extractor(rootpath, label_dir, npyfolder, data_type='train'):
  """
  :param rootpath: directory of input images
  :param label_dir: directory of ground-truth polygons in GeoJSON format
  :param npyfolder: folder to save the field data for each field polygon
  :param data_type: type of data train/test
  
  """
  with open(os.path.join(rootpath, "bbox.pkl"), 'rb') as f:
    bbox = pickle.load(f)
    crs = str(bbox.crs)
    minx, miny, maxx, maxy = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
  
  labels = gpd.read_file(label_dir)
  labels = labels.to_crs(crs)
    
  bands = np.load(os.path.join(rootpath, "bands.npy"))
  clp = np.load(os.path.join(rootpath, "clp.npy")) #CLOUD PROBABILITY
  
  _, width, height, _ = bands.shape
  transform = rio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

  bands=bands.transpose(0, 3, 1, 2)
  clp = clp.transpose(0, 3, 1, 2)

  fid_mask = features.rasterize(zip(labels.geometry, labels.fid), all_touched=True,
                            transform=transform, out_shape=(width, height))
  assert len(np.unique(fid_mask)) > 0, f"WARNING: Vectorized fid mask contains no fields. " \
                                        f"Does the label geojson {label_dir.split('/')[-1]} cover the region defined by {rootpath}?"

  crop_mask = features.rasterize(zip(labels.geometry, labels.crop_id), all_touched=True,
                            transform=transform, out_shape=(width, height))
  assert len(np.unique(crop_mask)) > 0, f"WARNING: Vectorized fid mask contains no fields. " \
                                        f"Does the label geojson {label_dir.split('/')[-1]} cover the region defined by {rootpath}?"

  for index, feature in tqdm(labels.iterrows(), total=len(labels), position=0, leave=True, desc="INFO: Extracting Sentinel-2 time series"):
    label = feature.crop_id
    fid = feature.fid
    #coords = feature.geometry.centroid.coords
    npyfile = f'{npyfolder}/{data_type}/s2/{label-1}/{fid}.npz'
    if data_type == 'test':
      npyfile = f'{npyfolder}/{data_type}/s2/{fid}.npz'
    if not os.path.exists(npyfile):
      left, bottom, right, top = feature.geometry.bounds
      window = rio.windows.from_bounds(left, bottom, right, top, transform)
      row_off = window.row_off
      col_off = window.col_off
      height = window.height
      width = window.width
      if row_off <0:
        #height = height + row_off
        row_off = 0
      if col_off <0:
        col_off = 0 
      row_start = math.floor(row_off)
      row_end = math.floor(row_off) + math.ceil(height)
      col_start = math.floor(col_off)
      col_end = math.floor(col_off) + math.ceil(width)
      image_stack = bands[:, :,row_start:row_end, col_start:col_end]
      cloud_stack =clp[:, :, row_start:row_end, col_start:col_end]

      mask = copy.deepcopy(fid_mask[row_start:row_end, col_start:col_end])
      mask[mask != feature.fid] = 0
      mask[mask == feature.fid] = 1
      os.makedirs(npyfolder, exist_ok=True)
      np.savez(npyfile, image_stack=image_stack.astype(np.float32), cloud_stack=cloud_stack.astype(np.float32), mask=mask.astype(np.float32), feature=feature.drop("geometry").to_dict())

def PlanetExtractor(rootpath, label_dir, npyfolder, tile, aoi='sa', data_type='train', planet='planet'):
  """
  :param rootpath: directory of input images
  :param label_dir: directory of ground-truth polygons in GeoJSON format
  :param npyfolder: folder to save the field data for each field polygon
  :param tile: tile name 
  :param aoi: Area of interest ('sa' for South Africa, 'germany' for Germany)
  :param data_type: type of data train/test
  """
  if aoi == 'sa':
    inputs = glob.glob(rootpath + f'/ref_fusion_competition_south_africa_{data_type}_source_{planet}_{tile}_*/sr.tif', recursive=True) #
  elif aoi == 'germany':
    inputs = glob.glob(rootpath + f'/dlr_fusion_competition_germany_{data_type}_source_{planet}_{tile}_*/sr.tif', recursive=True)
  tifs = sorted(inputs)
  #print(rootpath + f'/dlr_fusion_competition_germany_{data_type}_source_{planet}_{tile}_*/sr.tif')
  labels = gpd.read_file(label_dir)

  # read coordinate system of tifs and project labels to the same coordinate reference system (crs)
  with rio.open(tifs[1]) as image:
      crs = image.crs
      transform = image.transform
  
  labels = labels.to_crs(crs)
  
  for index, feature in tqdm(labels.iterrows(), total=len(labels), position=0, leave=True, desc="INFO: Extracting Planet time series"):
    label = feature.crop_id
    fid = feature.fid
    #coords = feature.geometry.centroid.coords
    npyfile = f'{npyfolder}/{data_type}/{planet}/{label-1}/{fid}.npz'
    if data_type == 'test':
      npyfile = f'{npyfolder}/{data_type}/{planet}/{fid}.npz'
    if not os.path.exists(npyfile):
      left, bottom, right, top = feature.geometry.bounds
      window = rio.windows.from_bounds(left, bottom, right, top, transform)
      row_off = window.row_off
      col_off = window.col_off
      height = window.height
      width = window.width
      if row_off <0:
        row_off = 0
      if col_off <0:
        col_off = 0

      window = Window(math.floor(col_off), math.floor(row_off), math.ceil(width), math.ceil(height)) 

      # reads each tif in tifs on the bounds of the feature. shape T x D x H x W
      image_stack = np.stack([rio.open(tif).read(window=window) for tif in tifs])

      with rio.open(tifs[0]) as src:
        win_transform = src.window_transform(window)

      out_shape = image_stack[0, 0].shape
      assert out_shape[0] > 0 and out_shape[1] > 0, "WARNING: fid:{} image stack shape {} is zero in one dimension".format(feature.fid,image_stack.shape)
      
      # rasterize polygon to get positions of field within crop
      mask = features.rasterize(feature.geometry, all_touched=True,transform=win_transform, out_shape=image_stack[0, 0].shape)

      os.makedirs(npyfolder, exist_ok=True)
      np.savez(npyfile, image_stack=image_stack, mask=mask, feature=feature.drop("geometry").to_dict())







