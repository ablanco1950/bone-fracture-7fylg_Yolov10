# https://github.com/ultralytics/yolov3/issues/1136
import torch
#def strip_optimizer(f='weights/last.pt'):  # from utils.utils import *; strip_optimizer() 
# Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
f="last20epoch25hits.pt"
f="last38epoch.pt"
x = torch.load(f, map_location=torch.device('cpu')) 
x['optimizer'] = None 
torch.save(x, f) 
