import torch
import torch.nn.functional as F 
from torch import nn 

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=100, z_dim=4) -> None:
        super().__init__()
        
        # This is for encoder
        self.img2hid = nn.Linear(input_dim, h_dim)
        self.hid2mu = nn.Linear(h_dim, z_dim)
        self.hid2logvar = nn.Linear(h_dim, z_dim)
        
        # This is for decoder
        self.z2hid = nn.Linear(z_dim, h_dim)
        self.hid2img = nn.Linear(h_dim, input_dim)
        
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        """_summary_

        Args:
            x (_type_): _description_
        """
        
        h = self.relu(self.img2hid(x))
        mu, logvar = self.hid2mu(h), self.hid2logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        """_summary_

        Args:
            z (_type_): _description_
        """
        h = self.relu(self.z2hid(z))
        return torch.sigmoid(self.hid2img(h)) 
    

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_
        """
        mu, logvar = self.encode(x)
        z_repara = self.reparameterize(mu, logvar)
        x_recons = self.decode(z_repara)
        return x_recons, mu, logvar 
    
    
class VAE_LOSS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
        

