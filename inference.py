import os
import sys
from random import randint
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.notebook import tqdm
import time
import gc
from datetime import datetime
#from utils import PyTorchSparkDataset, SPARKDataset
import csv
import cv2
import json
from itertools import chain
from statistics import mean


MODEL_PATH = "./Outputs/outputs-2023-01-04-12:15:46/model-afrantz-2023-01-04-12:15:46.pth"
COLORS = np.random.uniform(0, 255, size=(11, 3))
IMAGE_DIM = (1024, 1024)
class_map = {
    '__background__':          0,
    'proba_2':                 1,
    'cheops':                  2,
    'debris':                  3,
    'double_star':             4,
    'earth_observation_sat_1': 5,
    'lisa_pathfinder':         6,
    'proba_3_csc':             7,
    'proba_3_ocs':             8,
    'smart_1':                 9,
    'soho':                    10,
    'xmm_newton':              11,
} 
class_map_list = [
        '__background__',
        'proba_2',
        'cheops',
        'debris',
        'double_star',
        'earth_observation_sat_1',
        'lisa_pathfinder',
        'proba_3_csc',
        'proba_3_ocs',
        'smart_1',
        'soho',
        'xmm_newton'
        ] # Class map




# Load Model


def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_prediction(img_path, MODEL_PATH, threshold):
  """
  get_prediction
    parameters:
      - img_path - path of the input image
      - threshold - threshold value for prediction score
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - class, box coordinates are obtained, but only prediction score > threshold
        are chosen.
     
  """
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  # Load Model and Saved Weights
  model = create_model(12)
  # Set Device
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  # Load Weights
  state_dict = torch.load(MODEL_PATH, map_location=device)
  model.load_state_dict(state_dict, strict=True)
  # Set Model to eval mode
  model.eval()
  with torch.no_grad():
    pred = model([img])
    #print(f'pred : {pred}')
    #print(f'pred_0 : {pred[0]}')
    pred_class = [class_map_list[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    try:
      pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
      pred_boxes = pred_boxes[:pred_t+1]
      pred_class = pred_class[:pred_t+1]
      return pred_boxes, pred_class
    except:
      print("predictions below threshold")
      return 0, 0

def bb_intersection_over_union(boxA, boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  print(f'pred_box input = {boxA}')
  print(f'pred_box input type = {type(boxA)}')
  print(f'gt box input = {boxB}')
  print(f'gt box input type = {type(boxB)}')
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
  interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
  return iou

COLORS = np.random.uniform(0, 255, size=(11, 3))

# Load
#model = create_model(12)
#state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
#model.load_state_dict(state_dict, strict=True)
#model.eval()

class_map = {
    '__background__':          0,
    'proba_2':                 1,
    'cheops':                  2,
    'debris':                  3,
    'double_star':             4,
    'earth_observation_sat_1': 5,
    'lisa_pathfinder':         6,
    'proba_3_csc':             7,
    'proba_3_ocs':             8,
    'smart_1':                 9,
    'soho':                    10,
    'xmm_newton':              11,
} 

class_map_list = [
        '__background__',
        'proba_2',
        'cheops',
        'debris',
        'double_star',
        'earth_observation_sat_1',
        'lisa_pathfinder',
        'proba_3_csc',
        'proba_3_ocs',
        'smart_1',
        'soho',
        'xmm_newton'
        ] # Class map


IMAGE_DIM = (1024, 1024)

def inference_main(MODEL_PATH, threshold, USE_SMALL_DATASET=True):
  # Load Val dataset
  if USE_SMALL_DATASET:
    print("Using small dataset.")
    image_path = "/work/projects/spark2022-alks/dataset-small/Stream-1/val/"
    #image_path = "/content/data/val/"
  else:
    print("Using full dataset.")
    image_path = "/work/projects/spark2022-alks/dataset/Stream-1/val/"

  # Load val.csv
  if USE_SMALL_DATASET:
    print("Using small dataset.")
    val_csv = "/work/projects/spark2022-alks/dataset-small/Stream-1/val.csv"
    #val_csv = "/content/val.csv"
  else:
    print("Using full dataset.")
    val_csv = "/work/projects/spark2022-alks/dataset/Stream-1/val.csv"

  # Run inference on val dataset
  image_column = []
  pred_bbox_column = []
  gt_bbox_column = []
  pred_class_column = []
  gt_class_column = []
  iou_column = []
  for images in os.listdir(image_path):
    print("for loop")
    print(images)
    image_column.append(images) # adding entry for image
    # check if the image ends with png
    if (images.endswith(".png")):
      img_path = image_path + images
      #try:
      pred_box, pred_class = get_prediction(img_path, MODEL_PATH, threshold)
      #print(f'Pred_Box = {pred_box}')
      #print(f'Pred box type = {type(pred_box)}')
      print(f'Pred_Class = {pred_class}')
      #object_detection_api(img_path, threshold) 
      # Get ground truth data for corresponding image
      image_name = os.path.basename(images)
      print(image_name)
      val_csv_data = pd.read_csv(val_csv)
      filename = 'filename'
      clss = 'class'
      bbox = 'bbox'
      df_row = val_csv_data.loc[val_csv_data[filename] == image_name]
      gt_bbox = df_row[bbox].values[0]
      gt_label = df_row[clss].values[0]
      #print(val_csv_data)
      print(f'gt_bbox : {gt_bbox}')
      gt_bbox = json.loads(gt_bbox)
      #print(gt_label)
      if pred_box != 0:
        pred_box = [val for sublist in pred_box for tup in sublist for val in tup]
        print(f'Pred_Box new = {pred_box}')
        print(f'Pred box new type = {type(pred_box)}')
        iou = bb_intersection_over_union(pred_box, gt_bbox)
        print(f'IOU = {iou}')

        # Display images with pred_bbox
        img = cv2.imread(img_path)

        # Read image with cv2 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.copy()
        color = COLORS[class_map_list.index(pred_class[0])]
        cv2.rectangle(img, (int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])), color, 2) 
        # Draw Rectangle with the coordinates 
        cv2.putText(img, str(pred_class[0]), (int(pred_box[0]), ((int(pred_box[1]))-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, lineType=cv2.LINE_AA) 
        # Write the prediction class 
        plt.figure(figsize=(20,30)) 
        # display the output image 
        plt.imshow(img) 
        plt.xticks([]) 
        plt.yticks([]) 
        plt.show()

      else:
        iou = 0
      pred_bbox_column.append(pred_box)
      gt_bbox_column.append(gt_bbox)
      iou_column.append(iou)
      pred_class_column.append(pred_class)
      gt_class_column.append(gt_label)

  # Calculate Average IoU
  avg_iou = mean(iou_column)
  print(f'The Average IoU is {avg_iou}')

  # Create a pandas dataframe
  pd_data = {'Image':image_column, 'Pred_class':pred_class_column, 'gt_class':gt_class_column, 'Pred_bbox':pred_bbox_column, 'GT_bbox':gt_bbox_column, 'IoU':iou_column}
  df = pd.DataFrame(pd_data)
  print(df)

  # Save results to a csv
  # Get current date and time
  now = datetime.now()

  # Format date and time as string
  date_time = now.strftime("%Y-%m-%d %H-%M-%S")

  # Create file name with current date and time
  file_name = "inference_results_" + date_time + ".csv"

  # Write DataFrame to CSV file
  df.to_csv(file_name, index=False)