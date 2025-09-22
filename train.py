from train_func import train_valid
from model import main_model
import torch
import S_to_Graph
from predict import predict_from_trained_models_2
from fold_change import calc_log_fc
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from data_process import data_processing
from data_process import clean_and_convert_to_array
from r2_compounds import calc_r2_in_compounds
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU")
krags={
    "in_channels" : 79, 
    "hidden_channels" : 64, 
    "out_channels" : 64, 
    "edge_dim" : 10, 
    "num_layers" : 3, 
    "num_timesteps" : 3,
    "z_dim" :64, 
    "x_dim" : 978, 
    "num_embeddings" : 1024, 
    "num_epochs" : 500, 
    "patience" : 30, 
    "weight_decay" : 1e-6, 
    "lr" : 0.001, 
    "batchsize" : 32768,
    "kl_beta" : 0
}

model_path = rf'model_path/model.pth'
train_valid(krags,traindata_path=rf'data/train.tsv',
            model_save_path= model_path,
            checkpoint_path = rf'model_path/checkpoint.pth',
            valid=True,validdata_path=rf'data/valid.tsv',
            continue_train=False)




