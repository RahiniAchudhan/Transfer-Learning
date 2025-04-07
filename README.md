# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model.
</br>

## DESIGN STEPS
### STEP 1:
Import required libraries, load the dataset, and define training & testing datasets.
</br>

### STEP 2:
Initialize the model, loss function, and optimizer. Use CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.
</br>

### STEP 3:
Train the model using the training dataset with forward and backward propagation.
<br/>

### STEP 4:
Train the model using the training dataset with forward and backward propagation.
<br/>

### STEP 5:
Make predictions on new data using the trained model.
<br/>

## PROGRAM
```python
# Load Pretrained Model and Modify for Transfer Learning
from torchvision.models import VGG19_Weights
model = models.vgg19(weights=VGG19_Weights.DEFAULT


# Modify the final fully connected layer to match the dataset classes
num_classes = len(train_dataset.classes)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features,1)


# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)



# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float() )
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:Rahini A")
    print("Register Number:212223230165")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2025-04-07 113943](https://github.com/user-attachments/assets/8f1cc569-4115-47eb-aa9a-751003e24fad)
</br>

### Confusion Matrix
![Screenshot 2025-04-07 114034](https://github.com/user-attachments/assets/a5b9e59e-7166-4e39-8c10-9f1829528706)
</br>

### Classification Report
![Screenshot 2025-04-07 114107](https://github.com/user-attachments/assets/eab00f1e-bd1b-453b-820a-e4987a012adc)
</br>

### New Sample Prediction
![Screenshot 2025-04-07 114152](https://github.com/user-attachments/assets/ed5c1292-983e-47d7-a0a9-df7fb855270f)
![Screenshot 2025-04-07 114158](https://github.com/user-attachments/assets/8f2ff651-c206-4197-9407-d2610d8c3648)
</br>

## RESULT
Thus, the Transfer Learning for classification using the VGG-19 architecture has been successfully implemented.
</br>
