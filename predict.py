from model import main_model
import torch
import S_to_Graph

import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from data_process import data_processing

if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")



def predict_from_trained_models(data,model):
    model.eval()
    data = data_processing(data)
    smlies = data['canonical_smiles']
    
    data_list = S_to_Graph.create_pytorch_geometric_graph_data_list_from_smiles_and_labels(
        smlies, data['pertubed_sequence'], data['unpertubed_sequence'])
    
    dataloader = DataLoader(data_list, batch_size=4096)
    all_predictions = []
    with torch.no_grad():
        for (k, batch) in enumerate(dataloader):  
            batchsize = max(batch.batch)+1
            batch=batch.to(device)
            unpertubed_sequence = batch.unpertubed_sequence.reshape(batchsize, -1)
            pertubed_sequence = batch.pertubed_sequence.reshape(batchsize, -1)


            output, _= model.predict(batch.x,batch.edge_index,batch.edge_attr,batch.batch,unpertubed_sequence,pertubed_sequence)
            
            output=output.detach().cpu().numpy()
            all_predictions.extend(output)

    pred_series = pd.Series(all_predictions, name='predicted_value')

    df = pred_series.to_frame()
    return df


def predict_from_trained_models_2(data_path,save_path,model,krags):
    model.eval()
    data = pd.read_csv(data_path,sep='\t')
    data = data_processing(data)
    smlies = data['canonical_smiles']
    data_list = S_to_Graph.create_pytorch_geometric_graph_data_list_from_smiles_and_labels(
        smlies, data['pertubed_sequence'], data['unpertubed_sequence'])
    dataloader = DataLoader(data_list, batch_size=krags['batchsize'])
    all_predictions = []
    avg_perplexity=0
    with torch.no_grad():
        for (k, batch) in enumerate(dataloader):  
            batchsize = max(batch.batch)+1
            batch=batch.to(device)

            unpertubed_sequence = batch.unpertubed_sequence.reshape(batchsize, -1)
            pertubed_sequence = batch.pertubed_sequence.reshape(batchsize, -1)



            output, _= model(batch.x,batch.edge_index,batch.edge_attr,batch.batch,unpertubed_sequence,pertubed_sequence)
            output=output.detach().cpu().numpy()
            all_predictions.extend(output)
    pred_series = pd.Series(all_predictions, name='predicted_value')
    pred_series.to_csv(save_path, sep='\t',index=False)
    print("reconstruct successfully")
    print("perplexity:",avg_perplexity)


if __name__=='__main__':
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
    model=main_model(krags)
    model_path=r"model_to_save/model.pth"
    model=model.to(device)

    model.load_state_dict(torch.load(model_path,weights_only=True))

    data_path = r'data/test.csv'
    save_path=r'data/pred_test.csv'
    predict_from_trained_models_2(data_path,save_path,model,krags)


