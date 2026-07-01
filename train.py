import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.dataset import DNASequenceDataset
from src.model import CNN_Attention
from datetime import datetime
from tqdm import tqdm


# Function to create the model directory if it doesn't exist
def create_model_directory(base_dir="models"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


# Function to save the model with timestamped filename
def save_model(model, base_dir="models"):
    # Create model directory if it doesn't exist
    create_model_directory(base_dir)

    # Generate a timestamp for the model filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"model_{timestamp}.pth"
    model_path = os.path.join(base_dir, model_filename)

    # Save the model state dictionary
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model_path


# Train the model
def train_model(train_csv, validate_csv, batch_size=128, input_size=4, output_size=2,
                num_epochs=50, device='cuda', lr=0.001, weight_decay=1e-5, lr_decay_gamma=0.9, lr_decay_step=5):
    # Load the training and validation data
    train_dataset = DNASequenceDataset(train_csv)
    val_dataset = DNASequenceDataset(validate_csv)

    # Set the number of CPU threads for PyTorch
    num_threads = 50
    torch.set_num_threads(num_threads)

    # DataLoader for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)

    # Initialize the CNN + Attention model
    model = CNN_Attention(input_size=input_size, output_size=output_size)
    model = model.to(device)

    # Use CrossEntropyLoss for binary classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler (step decay)
    # Fixed: Changed lr_decay_factor to lr_decay_gamma
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    best_val_accuracy = 0
    best_model_path = None

    # Start training
    print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_file = open("training_log.txt", "w")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Wrap the DataLoader with tqdm to show progress
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{num_epochs}") as tepoch:
            for sequences, labels in tepoch:
                sequences, labels = sequences.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update the tqdm description
                tepoch.set_postfix(loss=train_loss / len(tepoch), accuracy=correct / total)

        # Calculate accuracy for this epoch
        accuracy = correct / total

        # Print current time, epoch info, loss, and accuracy
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}")
        log_file.write(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss / len(train_loader):.4f}, Training Accuracy: {accuracy:.4f}\n")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate validation accuracy
        val_accuracy = correct / total
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")
        log_file.write(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}\n")
        log_file.flush()

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = save_model(model)
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")

        # Step the scheduler
        scheduler.step()

    log_file.close()

    # Return the path of the best model
    return best_model_path


def main():
    # Path to the training CSV file
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_train_csv> <path_to_validate_csv>")
        sys.exit(1)

    train_csv = sys.argv[1]
    validate_csv = sys.argv[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    
    # Train the model
    model_path = train_model(
        train_csv=train_csv,
        validate_csv=validate_csv,
        batch_size=256,
        input_size=4,
        output_size=2,
        num_epochs=150,
        device=device,
        lr=0.001,
        weight_decay=1e-5,
        lr_decay_gamma=0.9,
        lr_decay_step=5
    )

    print(f"Training complete. Best model saved at: {model_path}")


if __name__ == "__main__":
    main()