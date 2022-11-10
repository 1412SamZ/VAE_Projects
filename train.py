import torch
import shutil
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader 
from torch import optim, nn 
from torch.optim.lr_scheduler import StepLR
from model import VariationalAutoEncoder as VAE, VAE_LOSS 
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
# experiment configurations

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(DEVICE)

INPUT_DIM = 28*28
H_DIM = 200
Z_DIM = 20

NUM_EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3

def show_img(img):
    img = img.permute(1, 2, 0)
    if img.shape[2]==1:
        img = img.view(img.shape[0], img.shape[1])
    plt.title(f'Image has size {img.cpu().numpy().shape}')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def check_img():
    for i in train_loader:
        original_image = i[0][0]
        show_img(original_image)
        
        break

# dataset
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = VAE(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

loss_fn = VAE_LOSS()


def evaluate(evaluate_data=valid_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(evaluate_data):
            
            data = data.resize(BATCH_SIZE, 28*28).to(DEVICE)
            
            recon_batch, mu, logvar = model(data)
            
            val_loss += loss_fn(recon_batch, data, mu, logvar).item()
            
            if i == 0:
                n = min(data.size(0), 16)
                comparison = torch.cat([data.view(BATCH_SIZE, 1, 28, 28)[:n],
                                      recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'Results/reconstruction_' + str(NUM_EPOCHS) + '.png', nrow=n)

    val_loss /= len(evaluate_data.dataset)
    return val_loss

# sample the data from a random latent space
def sample_latent_space(epoch):
    with torch.no_grad():
        sample = torch.randn(BATCH_SIZE, 20).to(DEVICE)
        sample = model.decode(sample).cpu()
        save_image(sample.view(BATCH_SIZE, 1, 28, 28),
                   'Results/sample_' + str(epoch) + '.png')
        
        
def train(epoch):

    model.train()
    train_loss = 0
    
    progress_bar = tqdm(train_loader, desc='Epoch {:03d}'.format(epoch), leave=False, disable=False)
    for data, _ in progress_bar:
        
        data = data.resize(BATCH_SIZE, 28*28).to(DEVICE)

        
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        
        loss = loss_fn(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        
        optimizer.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(data))})

    average_train_loss = train_loss / len(train_loader.dataset)
    tqdm.write('Training set loss (average, epoch {:03d}): {:.3f}'.format(epoch, average_train_loss))
    val_loss = evaluate(valid_loader)
    tqdm.write('\t\t\t\t====> Validation set loss: {:.3f}'.format(val_loss))

    train_losses.append(average_train_loss)
    val_losses.append(val_loss)
    scheduler.step()
    if epoch%50==0:
        torch.save(model.state_dict(), f'Models/epoch_{epoch}.model')
        
train_losses = []
val_losses = []

for epoch in range(1, NUM_EPOCHS+1):
    train(epoch)
    sample_latent_space(epoch)
    
np.savetxt('Models/training_losses.txt', np.array(train_losses), delimiter='\n')
np.savetxt('Models/validation_losses.txt', np.array(val_losses), delimiter='\n')
