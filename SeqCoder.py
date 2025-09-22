import torch
import torch.nn as nn


out_channels=64

if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

class SeqCoder(nn.Module):
    def __init__(self, x_dim, z_dim,out_channels,**kwargs):

        super(SeqCoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.lr = kwargs.get("learning_rate", 0.001)
        self.dr_rate = kwargs.get("dropout_rate", 0.3)


        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 700),
            nn.BatchNorm1d(700),
            nn.ReLU(),
            nn.Linear(700, 400),
            nn.Dropout(self.dr_rate),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            
        )


        self.mu_layer = nn.Linear(400, z_dim)       
        self.logvar_layer = nn.Linear(400, z_dim)  


        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)   
        return mu + eps * std         

    def forward(self, x):

        enc_output = self.encoder(x)

        mu = self.mu_layer(enc_output)
        logvar = self.logvar_layer(enc_output)

        z = self.reparameterize(mu, logvar)

        return  z,mu,logvar
    
class Decoder(nn.Module):
    def __init__(self, z_dim,x_dim,out_channels,dr_rate=0.5):

        super(Decoder, self).__init__()
        self.dr_rate = dr_rate
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + out_channels, 400),
            
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 700),
            nn.BatchNorm1d(700),
            nn.ReLU(),
            nn.Dropout(self.dr_rate),
            nn.Linear(700, x_dim),
            nn.LeakyReLU()
        )
    def forward(self, x):
        dec_output = self.decoder(x)
        return dec_output

