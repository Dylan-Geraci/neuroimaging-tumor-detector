"""
Quick test script to verify training works with minimal data.
"""
import torch
from src.data import get_data_loaders
from src.model import create_model
from src.train import Trainer

print("=" * 50)
print("QUICK TRAINING TEST")
print("=" * 50)

# Load data with very small batch
print("\n1. Loading data (batch_size=4)...")
train_loader, val_loader, test_loader = get_data_loaders(
    data_dir='data',
    batch_size=4,
    num_workers=0  
)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# Create model
print("\n2. Creating model...")
model = create_model(num_classes=4, pretrained=False, device='cpu')  # No pretrained for speed
device = next(model.parameters()).device
print(f"   Device: {device}")

# Test single forward pass
print("\n3. Testing forward pass...")
images, labels = next(iter(train_loader))
images, labels = images.to(device), labels.to(device)
print(f"   Input shape: {images.shape}")

with torch.no_grad():
    outputs = model(images)
    print(f"   Output shape: {outputs.shape}")
    print("   ✓ Forward pass successful!")

# Test single training step
print("\n4. Testing single training step...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

model.train()
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
print(f"   Loss: {loss.item():.4f}")
print("   ✓ Training step successful!")

# Try 1 epoch with limited batches
print("\n5. Running 1 mini-epoch (10 batches)...")
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=str(device),
    learning_rate=0.001,
    checkpoint_dir='models'
)

# Manually run a few batches
model.train()
batch_count = 0
for images, labels in train_loader:
    if batch_count >= 10:  # Only 10 batches
        break

    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    batch_count += 1
    print(f"   Batch {batch_count}/10 - Loss: {loss.item():.4f}")

print("\n" + "=" * 50)
print("TEST COMPLETE - Everything works!")
print("=" * 50)
