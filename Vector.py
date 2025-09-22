import torch
import torch.nn as nn
if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, divergence_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
        self._divergence_cost = divergence_cost

    def forward(self, inputs): 

            
        distances = torch.cdist(inputs, self._embedding.weight)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
        encodings.scatter_(1, encoding_indices, 1)
        
  
        quantized = torch.matmul(encodings, self._embedding.weight)
        
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss*self._divergence_cost+ self._commitment_cost * e_latent_loss    
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss
    