import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.s4.s4_block import S4Block 

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

# Hyperparameters
batch_size = 64
sequence_length = 784  # 28x28 pixels
d_model = 128 # Model dimension
num_classes = 10
learning_rate = 1e-3 #1e-4
num_epochs = 10
num_workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir =  'data/'

train_dataset = torchvision.datasets.MNIST(root = data_dir,
                                           train = True, 
                                           transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
                                           download = True)

test_dataset = torchvision.datasets.MNIST(root =  data_dir,
                                          train = False, 
                                          transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]))


loss_f = nn.CrossEntropyLoss()

# we drop the last batch to ensure each batch has the same size
train_loader = DataLoader(dataset = train_dataset, num_workers = num_workers,
                                            batch_size = batch_size,
                                            shuffle = True, drop_last = False)
        
test_loader = DataLoader(dataset = test_dataset, num_workers = num_workers,
                                            batch_size = batch_size,
                                            shuffle = False, drop_last = False)

# Initialize the S4Block model
encoder = nn.Linear(1, d_model).to(device)
model = S4Block(d_model=d_model, dropout=0.2, transposed=False).to(device)
layernorm = nn.LayerNorm(d_model).to(device)
dropout = dropout_fn(0.2).to(device)
# Initialize dA, dB, and other state-related attributes
# needed for recurrent mode
model.setup_step()  
decoder = nn.Linear(d_model, num_classes).to(device)

print(encoder)
print(model)
print(layernorm)
print(dropout)
print(decoder)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.parameters()) + 
                       list(encoder.parameters()) + 
                       list(decoder.parameters()) +
                       list(layernorm.parameters()), lr=learning_rate,  weight_decay = 0.00, betas = (0.9, 0.999))

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        #print("x shape:", x.shape)
        #x = x.view(batchsize, sequence_length, 1).to(device)
        x = x.view(x.shape[0], -1, 1).to(device)
        #print("x shape:", x.shape)
        
        y = y.to(device)

        # Initialize state
        state = model.default_state(x.shape[0], device=device)

        z = x
        # Pass the input through the encoder
        # Shape: (batch_size, sequence_length, d_model)
        #print("x.shape:", x.shape)
        x = encoder(x)
        #print("encoder shape:", x.shape)
        x = x.transpose(-1, -2)
        #print("x transpose 1 shape:", x.shape)
        #for t in range(sequence_length):
        z_list = []
        #print(x.shape[-1])
        for t in range(x.shape[-1]):

            #x_t = x[:, t, :]
            x_t = x[:, :, t]
            #print("x_t.shape:", x_t.shape)
            #x_t = encoder(x_t) 
            z, state = model.step(x_t, state)
            #print("z.shape:", z.shape)
            z_list.append(z)
        
        #print("len(z_list):", len(z_list))
        z = torch.stack(z_list, dim=-1)
        #print("torch.stack(z_list)", z.shape)
        #z = layernorm(z)
        z = dropout(z)
        #print("z dropout shape:", z.shape)
        
        #print("Pre resid z.shape:", z.shape)
        #print("Pre resid x.shape:", x.shape)
        x = z + x
        #print("residual shape:", x.shape)

        x = layernorm(x.transpose(-1, -2)).transpose(-1, -2)
        #print("not prenorm x.shape:", x.shape)
        
        #print("x transpose 2 shape:", x.shape)

        x = x.transpose(-1, -2)
        x = x.mean(dim=1)
        #print("avg pool shape:", x.shape)

        logits = decoder(x)
        #print("decoder shape:", x.shape)

        loss = criterion(logits, y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Evaluation Phase
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x, y in test_loader:  # Use validation/test loader
            x = x.view(x.shape[0], -1, 1).to(device)
            y = y.to(device)

            # Initialize state
            state = model.default_state(x.shape[0], device=device)

            # Encode input
            x = encoder(x)
            x = x.transpose(-1, -2)

            # Process sequence
            z_list = []
            for t in range(x.shape[-1]):
                x_t = x[:, :, t]
                z, state = model.step(x_t, state)
                z_list.append(z)

            z = torch.stack(z_list, dim=-1)
            z = dropout(z)

            # Residual connection + LayerNorm
            x = z + x
            x = layernorm(x.transpose(-1, -2)).transpose(-1, -2)

            # Pooling
            x = x.transpose(-1, -2).mean(dim=1)

            # Decode
            logits = decoder(x)

            # Compute loss
            loss = criterion(logits, y)
            test_loss += loss.item()

            # Accuracy
            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total
    print(f"Validation Loss: {avg_test_loss:.4f}, Validation Accuracy: {test_accuracy:.2f}%")