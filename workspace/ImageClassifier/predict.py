import torch
import argparse
from utils.predict_fns import load_model
#from utils.training_fns import train_model, load_model
#from utils.load_data import load_data
#from torch.optim import Adam
#import torch.nn as nn

parser =argparse.ArgumentParser(description= 'Predict the class of an Image.')
parser.add_argument('image_path')
parser.add_argument('--model_path', default="gpu")

args = parser.parse_args()

model=load_model(args.model_path)

predict_and_plot(args.image_path,model)