# DrugPerturb

We propose a model named CA-VAE. This framework incorporates a self-attention-based molecular graph encoder(AttentiveFP)  to extract chemical features from SMILES representations of compounds,enabling generalization to unseen compounds without prior knowledge and annotation. The model further refines drug feature vectors through a self-trained CODEBOOK for clustering optimization, and ultimately employs a Variational Autoencoder (VAE) to generate perturbed sequential features.
