from model import main_model
import torch
import S_to_Graph 
import pandas as pd
from torch_geometric.loader import DataLoader
from data_process import data_processing
import matplotlib.pyplot as plt
import os
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU")

def train_valid(krags, traindata_path, model_save_path, checkpoint_path=None, validdata_path=None, training=True, valid=False, continue_train=False):
    train_data=traindata_path
    train_data= pd.read_csv(train_data,sep='\t')
    train_data=data_processing(train_data)
    print("train data processed successfully!")
    smlies=train_data['canonical_smiles']
    data_list=S_to_Graph.create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smlies, train_data['perturbed_sequence'],train_data['unperturbed_sequence'])
    dataloader = DataLoader(data_list , batch_size = krags['batchsize'])
    train_losses = []
    if valid:
        valid_data= pd.read_csv(validdata_path,sep='\t')
        valid_data=data_processing(valid_data)
        print("valid data processed successfully!")
        valid_smlies=valid_data['canonical_smiles']
        valid_data_list=S_to_Graph.create_pytorch_geometric_graph_data_list_from_smiles_and_labels(valid_smlies, valid_data['perturbed_sequence'],valid_data['unperturbed_sequence'])
        dataloader2 = DataLoader(valid_data_list , batch_size = krags['batchsize'])
        valid_losses = []
        
        best_val_loss = float('inf')
        no_improve_epochs = 0  
    


    model=main_model(krags)
    model=model.to(device)   
    optimizer1 = torch.optim.Adam(model.parameters(), lr=krags["lr"], weight_decay=krags["weight_decay"])
    start_epoch = 0

    if continue_train and checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path,weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer1.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint.get("train_losses", [])
        if valid:
            valid_losses = checkpoint.get("valid_losses", [])
            best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        print(f"Resuming training from epoch {start_epoch}")    
    num_epochs = krags["num_epochs"]
    for epoch in range(start_epoch, num_epochs):
        total_loss=0
        avg_perplexity=0
        model.train()
        for (k,batch) in enumerate(dataloader):
            

            batch=batch.to(device)
            #print(batch)
            batchsize = max(batch.batch)+1

            unperturbed_sequence = batch.unperturbed_sequence.reshape(batchsize,-1)
            perturbed_sequence = batch.perturbed_sequence.reshape(batchsize,-1)

            
            _, loss = model(batch.x,batch.edge_index,batch.edge_attr,batch.batch,unperturbed_sequence,perturbed_sequence)
        
            optimizer1.zero_grad()

            loss.backward()
            optimizer1.step()
            total_loss+=loss.item()
        avg_train_loss = total_loss / len(dataloader)
        print(f'[{epoch}/{num_epochs}]\nTrain_loss: {avg_train_loss}')
        train_losses.append(avg_train_loss)
    
        if valid:

            model.eval()

            total_val_loss = 0

            with torch.no_grad():
                for (k, batch) in enumerate(dataloader2):  

                    batch=batch.to(device)
                    #print(batch)
                    batchsize = max(batch.batch)+1

                    unperturbed_sequence = batch.unperturbed_sequence.reshape(batchsize,-1)
                    perturbed_sequence = batch.perturbed_sequence.reshape(batchsize,-1)

                    
                    _,valid_loss= model(batch.x,batch.edge_index,batch.edge_attr,batch.batch,unperturbed_sequence,perturbed_sequence)


                    total_val_loss+=valid_loss.item()
                avg_valid_loss = total_val_loss / len(dataloader2)
                valid_losses.append(avg_valid_loss)

                print(f'Valid_loss: {avg_valid_loss}\n')
                if avg_valid_loss < best_val_loss:
                    best_val_loss = avg_valid_loss
                    no_improve_epochs = 0
                    torch.save(model.state_dict(), model_save_path)
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= krags["patience"]:
                    print(f'Early stopping after {epoch+1} epochs without improvement.')
                    break
        checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer1.state_dict(),
                "train_losses": train_losses,
                "valid_losses": valid_losses if valid else None,
                "best_val_loss": best_val_loss if valid else None
            }
        torch.save(checkpoint, checkpoint_path)        
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if valid:
        plt.plot(valid_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/loss_curve.png')
    plt.show()

    print("Models saved successfully!")