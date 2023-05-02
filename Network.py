import numpy as np
import torch
import torchvision.transforms.functional as F

# Define the Lasso model
class Lasso(torch.nn.Module):
    def __init__(self, num_sparce_vector, number_of_atoms):
        super(Lasso, self).__init__()
        self.sparse_vector = torch.nn.Parameter(torch.randn(number_of_atoms, num_sparce_vector, requires_grad=True))

    def forward(self, x):
        return x.T @ self.sparse_vector
    
    def fit(self, patch, dictionary, lr = 0.03, num_epochs = 6000, alpha=0.1, betas =(0.9,0.999)):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas= betas)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        loss = torch.nn.MSELoss()

        for epoch in range(num_epochs):

            # Forward pass
            predict = self.forward(dictionary)

            # L1-Regularization
            l1_penalty = alpha * sum(torch.linalg.norm(p, 1) for p in self.parameters())
            err = loss(predict, patch) + l1_penalty
            if epoch%10 == 0:
                print(f"loss of epoch {epoch} => {np.round(err.detach().numpy(),2)}")
            # Backward pass and optimization
            optimizer.zero_grad()
            err.backward()
            optimizer.step()