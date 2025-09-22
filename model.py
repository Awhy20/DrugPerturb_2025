import torch
import torch.nn as nn
from Attentive_fp import AttentiveFP
from SeqCoder import SeqCoder
from SeqCoder import Decoder
from Vector import VectorQuantizer
import logging
from torch.nn import GRUCell
from torch_geometric.nn import GATConv
log = logging.getLogger(__file__)


if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

    
class main_model(nn.Module):

    def __init__(self,krags):
        super(main_model, self).__init__()

        self.mol_encoder = AttentiveFP(
        out_channels=krags["out_channels"],
        in_channels=krags["in_channels"], 
        hidden_channels=krags["hidden_channels"],
        edge_dim=krags["edge_dim"], 
        num_layers=krags["num_layers"], 
        num_timesteps=krags["num_timesteps"],
        training=True, 
        dropout=0.3
        )

        self.seq_encoder = SeqCoder(x_dim=krags["x_dim"],out_channels=krags['out_channels'],z_dim=krags["z_dim"],learning_rate=krags["lr"])
        self.vector_quantizer = VectorQuantizer(num_embeddings=krags["num_embeddings"], embedding_dim=krags['out_channels'],commitment_cost=0.25,divergence_cost=0.05)
        self.decoder=Decoder(z_dim=krags["z_dim"],x_dim=krags['x_dim'],out_channels=krags['out_channels'])
        self._initialize_parameters()
        self.kl_beta=krags['kl_beta']
    def forward(self, x, edge_index, edge_attr, batch, unpert_sequence, pert_sequence):
        
        mol_ft = self.mol_encoder(x, edge_index, edge_attr, batch)
        mol_ft2, vector_loss = self.vector_quantizer(mol_ft)
        seq_ft, mu, logvar = self.seq_encoder(unpert_sequence)
        dec_input = torch.cat([mol_ft2, seq_ft], dim=1)
        dec_output = self.decoder(dec_input)

        recon_loss = self.recon_loss(dec_output, pert_sequence, mu, logvar)
        loss = recon_loss + vector_loss
        return dec_output, loss

 


    def predict(self, x, edge_index, edge_attr, batch, unpert_sequence, pert_sequence):
        mol_ft = self.mol_encoder(x, edge_index, edge_attr, batch)
        mol_ft2, vec_loss= self.vector_quantizer(mol_ft)
        seq_ft, mu, logvar = self.seq_encoder(unpert_sequence)
        dec_input = torch.cat([mol_ft2, seq_ft], dim=1)
        dec_output = self.decoder(dec_input)

        recon_loss = self.recon_loss(dec_output, pert_sequence, mu, logvar)
        loss = recon_loss + vec_loss
        return dec_output, loss

    def recon_loss(self, dec_output, real_y, mu, logvar):
        recon_loss = torch.sum((dec_output - real_y)**2, dim=1)
        recon_loss = torch.mean(recon_loss)

        total_loss = recon_loss 
        return total_loss

    def kl_divergence(self, mu, logvar):

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return torch.mean(kl_loss)
    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=1)  
            elif isinstance(module, GRUCell):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:  
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:  
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

            elif isinstance(module, nn.BatchNorm1d):

                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

