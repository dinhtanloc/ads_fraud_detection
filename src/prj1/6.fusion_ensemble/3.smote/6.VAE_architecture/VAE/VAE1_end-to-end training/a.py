import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_Tabnet_MLPS(nn.Module):
    def __init__(self, encoder, decoder, classifier):
        super(VAE_Tabnet_MLPS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.bn = nn.BatchNorm1d(encoder_output_dim)  # Thay encoder_output_dim bằng kích thước đầu ra của encoder
        self.do = nn.Dropout(0.25)
        self.total_loss_tracker = []
        self.reconstruction_loss_tracker = []
        self.kl_loss_tracker = []
        self.classification_loss_tracker = []
        self.accuracy_tracker = []

    def forward(self, x):
        x = self.bn(x)
        x = self.do(x)
        
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        classification_output = self.classifier(z)
        
        return reconstruction, z_mean, z_log_var, classification_output

    def train_step(self, data, labels, optimizer):
        optimizer.zero_grad()
        reconstruction, z_mean, z_log_var, classification_output = self.forward(data)

        # Calculate losses
        reconstruction_loss = F.binary_cross_entropy(reconstruction, data, reduction='sum') / data.size(0)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / data.size(0)
        classification_loss = F.binary_cross_entropy(classification_output, labels, reduction='sum') / data.size(0)

        total_loss = reconstruction_loss + kl_loss + classification_loss
        total_loss.backward()
        optimizer.step()

        # Track metrics
        self.total_loss_tracker.append(total_loss.item())
        self.reconstruction_loss_tracker.append(reconstruction_loss.item())
        self.kl_loss_tracker.append(kl_loss.item())
        self.classification_loss_tracker.append(classification_loss.item())
        
        # Calculate accuracy
        preds = torch.sigmoid(classification_output)
        correct = ((preds > 0.5) == labels).float().sum()
        accuracy = correct / labels.size(0)
        self.accuracy_tracker.append(accuracy.item())

        return {
            "loss": total_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kl_loss": kl_loss.item(),
            "classification_loss": classification_loss.item(),
            "accuracy": accuracy.item()
        }
