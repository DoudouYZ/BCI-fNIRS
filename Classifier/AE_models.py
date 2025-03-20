import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------------
# Helper: Cropping1D Module
# -------------------------------
class Cropping1D(nn.Module):
    def __init__(self, crop_right):
        """
        Crops the last 'crop_right' time steps from the input.
        """
        super(Cropping1D, self).__init__()
        self.crop_right = crop_right

    def forward(self, x):
        # x shape: (batch, channels, time)
        if self.crop_right > 0:
            return x[:, :, :-self.crop_right]
        return x

# -------------------------------
# Encoder Module
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_channels, input_length, latent_dim):
        """
        Args:
            input_channels (int): Number of input channels (e.g. 40).
            input_length (int): Number of time steps (e.g. 79).
            latent_dim (int): Dimension of the latent representation.
        """
        super(Encoder, self).__init__()
        # First convolution: keep time dimension same
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32,
                               kernel_size=(3,1), stride=1, padding=1)
        # Second convolution: downsample by factor 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32,
                               kernel_size=(3,1), stride=1, padding=1)
        # Third convolution: further downsample by factor 2
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16,
                               kernel_size=(3,1), stride=1, padding=1)
        
        # Compute the output length after conv layers using the formula:
        # L_out = floor((L + 2*padding - kernel_size)/stride + 1)
        def conv_out_length(L, kernel_size=(3,1), stride=1, padding=1):
            return (L + 2 * padding - kernel_size[0]) // stride + 1
        
        L1 = conv_out_length(input_length, kernel_size=(3,1), stride=1, padding=1)
        L2 = conv_out_length(L1, kernel_size=(3,1), stride=1, padding=1)
        L3 = conv_out_length(L2, kernel_size=(3,1), stride=1, padding=1)
        self.conv_output_length = L3  # save for the decoder
        
        self.flattened_size = 16 * L3
        self.fc = nn.Linear(self.flattened_size, latent_dim)
    
    def forward(self, x):
        # x: (batch, input_channels, input_length)
        x = F.relu(self.conv1(x))     # -> (batch, 32, L)
        x = F.relu(self.conv2(x))     # -> (batch, 32, L2)
        x = F.relu(self.conv3(x))     # -> (batch, 16, L3)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)    # flatten
        latent = self.fc(x)           # -> (batch, latent_dim)
        return latent

# -------------------------------
# Reconstruction Decoder Module
# -------------------------------
class ReconstructionDecoder(nn.Module):
    def __init__(self, latent_dim, flattened_size, conv_output_length,
                 output_channels, original_length):
        """
        Args:
            latent_dim (int): Dimension of latent vector.
            flattened_size (int): Size from encoder (e.g., 16 * L3).
            conv_output_length (int): L3 from the encoder.
            output_channels (int): Number of channels in the output (e.g., 40).
            original_length (int): Original number of time steps (e.g., 79).
        """
        super(ReconstructionDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, flattened_size)
        self.conv_output_length = conv_output_length
        self.output_channels = output_channels
        self.original_length = original_length

        # First upsampling block
        self.conv_up1 = nn.Conv1d(in_channels=16, out_channels=32,
                                  kernel_size=(3,1), stride=1, padding=1)
        # Second upsampling block
        self.conv_up2 = nn.Conv1d(in_channels=32, out_channels=32,
                                  kernel_size=(3,1), stride=1, padding=1)
        # Final reconstruction layer
        self.conv_final = nn.Conv1d(in_channels=32, out_channels=output_channels,
                                    kernel_size=(3,1), stride=1, padding=1)
        # After two upsampling operations, the time dimension becomes conv_output_length * 4.
        self.final_length = conv_output_length * 4
        self.crop_amount = self.final_length - original_length
        if self.crop_amount < 0:
            raise ValueError("Upsampled length is smaller than original length.")
        self.crop = Cropping1D(crop_right=self.crop_amount)

    def forward(self, z):
        # z: (batch, latent_dim)
        x = self.fc(z)  # -> (batch, flattened_size)
        batch_size = x.shape[0]
        # Reshape to (batch, 16, conv_output_length)
        x = x.view(batch_size, 16, self.conv_output_length)
        x = self.upsample1(x)                   # -> (batch, 16, conv_output_length*2)
        x = F.relu(self.conv_up1(x))            # -> (batch, 32, conv_output_length*2)
        x = self.upsample2(x)                   # -> (batch, 32, conv_output_length*4)
        x = F.relu(self.conv_up2(x))            # -> (batch, 32, conv_output_length*4)
        x = self.conv_final(x)                  # -> (batch, output_channels, conv_output_length*4)
        # Crop to original length if necessary
        if self.crop_amount > 0:
            x = self.crop(x)
        return x

# -------------------------------
# Reconstruction Autoencoder
# -------------------------------
class ReconstructionAutoencoder(nn.Module):
    def __init__(self, input_channels, input_length, latent_dim):
        """
        Combines the encoder and reconstruction decoder.
        """
        super(ReconstructionAutoencoder, self).__init__()
        self.encoder = Encoder(input_channels, input_length, latent_dim)
        self.decoder = ReconstructionDecoder(
            latent_dim=latent_dim,
            flattened_size=self.encoder.flattened_size,
            conv_output_length=self.encoder.conv_output_length,
            output_channels=input_channels,
            original_length=input_length
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
    def get_latent(self, x):
        return self.encoder(x)

# -------------------------------
# Classification Autoencoder
# -------------------------------
class ClassificationAutoencoder(nn.Module):
    def __init__(self, input_channels, input_length, latent_dim, num_classes):
        """
        Uses the same encoder as above but attaches a classification head.
        """
        super(ClassificationAutoencoder, self).__init__()
        self.encoder = Encoder(input_channels, input_length, latent_dim)
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        z = self.encoder(x)
        x_cls = F.relu(self.fc1(z))
        logits = self.fc2(x_cls)  # raw logits; use CrossEntropyLoss which applies softmax internally
        return logits
    
    def get_latent(self, x):
        return self.encoder(x)

# -------------------------------
# Training Functions
# -------------------------------
def train_reconstruction_autoencoder(model, train_loader, val_loader, epochs, device):
    """
    Training loop for the reconstruction autoencoder.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)  # input and target are the same
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Reconstruction AE Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    return history

def train_classification_autoencoder(model, train_loader, val_loader, epochs, device):
    """
    Training loop for the classification autoencoder.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)  # targets should be LongTensor (class indices)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        train_loss /= len(train_loader.dataset)
        train_acc = correct / total
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * inputs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Classification AE Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    return history
