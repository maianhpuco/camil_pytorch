import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import json
from sklearn.metrics import roc_auc_score 


def train_one_epoch(model, train_dataset, val_dataset, optimizer, loss_fn, device):
    """
    Trains the model for one epoch on individual samples (without batching) and validates it on the validation set.
    
    Args:
        model: The model to be trained.
        train_dataset: The dataset to load training data.
        val_dataset: The dataset to load validation data.
        optimizer: The optimizer to update model parameters.
        loss_fn: The loss function to compute the error.
        device: The device to train on ('cuda' or 'cpu').
        
    Returns:
        avg_train_loss: The average loss for the training epoch.
        train_accuracy: The accuracy for the training epoch.
        avg_val_loss: The average loss for the validation epoch.
        val_accuracy: The accuracy for the validation epoch.
        train_auc: The AUC for the training epoch.
        val_auc: The AUC for the validation epoch.
    """
    
    # Training Phase
    model.train()  # Set the model to training mode
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    all_train_preds = []
    all_train_labels = []

    for features, sparse_matrix, labels in train_dataset:
        
        features = features.to(device)
        sparse_matrix = sparse_matrix.to(device)
        labels = labels.to(device)
        # Print the shape of each tensor
        # print("Features shape:", features.shape)
        # print("Sparse matrix shape:", sparse_matrix.shape)
        # print("Labels shape:", labels.shape)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: Get model outputs
        predicted_prob, _, _ = model(features, sparse_matrix)
        # print(predicted_prob)
        # print("predict prob:", predicted_prob) 
        # Calculate loss (assuming outputs are logits)
        loss = loss_fn(predicted_prob, labels)

        # Backward pass: Compute gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track running loss
        running_train_loss += loss.item()
        # Calculate accuracy (for binary classification)
        predicted = predicted_prob > 0.5  # Convert logits to binary class predictions
        correct_train += (predicted == labels.unsqueeze(1)).sum().item()
        total_train += labels.size(0)
        
        # Collect predictions and labels for AUC calculation
        all_train_preds.extend(predicted_prob.detach().cpu().numpy().flatten())  # Flatten to 1D
        all_train_labels.extend(labels.detach().cpu().numpy().flatten())  # Flatten to 1D 
    # print("---------------- ")
    # print("all_train_preds", all_train_preds)
    # print("all_train_labels", all_train_labels)  
    
     
    # Calculate average loss and accuracy for training
    avg_train_loss = running_train_loss / len(train_dataset)
    train_accuracy = correct_train / total_train

    # print("all_train_labels", all_train_labels)
    # print("all_train_preds", all_train_preds)
    # Calculate AUC for training
    train_auc = roc_auc_score(all_train_labels, all_train_preds)

    # Validation Phase
    model.eval()  # Set the model to evaluation mode
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():  # Disable gradient computation during validation
        for features, sparse_matrix, labels in val_dataset:
            features = features.to(device)
            sparse_matrix = sparse_matrix.to(device)
            labels = labels.to(device)

            # Forward pass: Get model outputs
            outputs, _, _ = model(features, sparse_matrix)
            predicted_prob = torch.sigmoid(outputs)

            # Calculate loss
            loss = loss_fn(predicted_prob, labels)

            # Track running loss
            running_val_loss += loss.item()

            # Calculate accuracy (for binary classification)
            predicted = predicted_prob > 0.5  # Convert logits to binary class predictions
            correct_val += (predicted == labels.unsqueeze(1)).sum().item()
            total_val += labels.size(0)

            # Collect predictions and labels for AUC calculation
            all_val_preds.extend(predicted_prob.detach().cpu().numpy().flatten())  # Flatten to 1D
            all_val_labels.extend(labels.detach().cpu().numpy().flatten())  # Flatten to 1D  
    # print("--")
    # print("all_val_preds", all_val_preds)
    # print("all_val_labels", all_val_labels) 
    
    # Calculate average loss and accuracy for validation
    avg_val_loss = running_val_loss / len(val_dataset)
    val_accuracy = correct_val / total_val

    # Calculate AUC for validation
    val_auc = roc_auc_score(all_val_labels, all_val_preds)

    return avg_train_loss, train_accuracy, avg_val_loss, val_accuracy, train_auc, val_auc
 
def train(
    model, train_dataset, val_dataset, epochs=10, learning_rate=1e-3, device="cuda", save_path="best_model.pth", log_file=None):
    """
    Train the model for multiple epochs, saving the model with the best validation performance and logging the results.
    
    Args:
        model: The model to be trained.
        train_dataset: The dataset to load training data from.
        val_dataset: The dataset to load validation data from.
        epochs: Number of epochs to train.
        learning_rate: Learning rate for optimizer.
        device: The device to train on ('cuda' or 'cpu').
        save_path: Path where to save the model weights of the best epoch.
        log_file: Path to save the log file containing epoch results.
    """
    
    # Move model to the specified device (GPU/CPU)
    model.to(device)
    
    # Set up the optimizer (using Adam here)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function (using binary cross-entropy for binary classification)
    loss_fn = torch.nn.BCELoss()  # Use this if your model outputs raw logits
    
    best_val_accuracy = 0.0  # Track the best validation accuracy
    best_epoch = -1  # To store the epoch of the best model
    best_model_weights = None  # To store the best model's weights

    # List to store logs for each epoch
    log_data = []

    # Training loop over multiple epochs
    for epoch in range(epochs):
        # Train and validate for the current epoch
        avg_train_loss, train_accuracy, avg_val_loss, val_accuracy, train_auc, val_auc = train_one_epoch(
            model, train_dataset, val_dataset, optimizer, loss_fn, device)
        
        # Print results for the current epoch
        # Print results for the current epoch, including AUC for both train and validation
        print(f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train AUC: {train_auc:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")

        # Log results for the current epoch, including AUC
        if log_file is not None:
            log_data.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "train_auc": train_auc,  # Add train AUC
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_auc": val_auc   # Add validation AUC
            }) 
        
        # If the current epoch has better validation accuracy, save the model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_model_weights = model.state_dict()  # Save the model's state dict

    # After all epochs, save the best model's weights
    if best_model_weights is not None:
        torch.save(best_model_weights, save_path)
        print(f"Best model saved at epoch {best_epoch} with validation accuracy: {best_val_accuracy:.4f}")
    else:
        print("No model improvement detected.")
    
    # Save the log data to a JSON file
    if log_file is not None:
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=4)
        print(f"Training log saved to {log_file}")
    
    print("Training complete") 
 
 
def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}...")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Resuming from epoch {epoch}, last loss: {loss}")
        return model, optimizer, epoch, loss
    else:
        print("No checkpoint found, starting fresh.")
        return model, optimizer, 0, None


def evaluate(model, val_dataset, device, checkpoint_path=None):
    """
    Evaluate the model performance on the validation or test dataset.
    
    Args:
        model (torch.nn.Module): The trained model.
        val_dataset (torch.utils.data.Dataset): The validation or test dataset.
        device (torch.device): The device to evaluate the model on ('cpu' or 'cuda').
        checkpoint_path (str, optional): The path to the model checkpoint for loading weights. Default is None.
    
    Returns:
        None
    """
    # Load model weights if checkpoint path is provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("No checkpoint path provided. Evaluating with the current model state.")
    
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize lists for collecting predictions and labels
    all_preds = []
    all_labels = []

    # Iterate through the validation dataset
    with torch.no_grad():  # No gradient computation for evaluation
        for features, sparse_matrix, labels in val_dataset:
            features = features.to(device)
            sparse_matrix = sparse_matrix.to(device)
            labels = labels.to(device)
            
            # Forward pass: Get model outputs
            outputs, _, _ = model(features, sparse_matrix)
            predicted_probs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            
            # Collect predictions and labels
            all_preds.extend(predicted_probs.cpu().numpy().flatten())  # Flatten to 1D
            all_labels.extend(labels.cpu().numpy().flatten())  # Flatten to 1D
    
    # Convert lists to numpy arrays for evaluation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute evaluation metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(all_labels, all_preds > 0.5)  # Threshold at 0.5 for binary classification
    auc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds > 0.5)
    recall = recall_score(all_labels, all_preds > 0.5)
    f1 = f1_score(all_labels, all_preds > 0.5)

    # Print evaluation results
    print(f"Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
