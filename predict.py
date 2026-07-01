import torch
import numpy as np
from src.model import CNN_Attention
import sys
from datetime import datetime
import psutil  # To track memory usage
from src.dataset import DNASequenceDataset


current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"{current_time}: Selecting chloroplast reads")


def get_memory_usage():
    process = psutil.Process()  # Get the current process
    return process.memory_info().rss / (1024 * 1024)  # Memory in MB


# Load the trained model
def load_model(model_path, input_size=4, output_size=2, device='cuda'):
    model = CNN_Attention(input_size=input_size, output_size=output_size)
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # Ignore unexpected keys and load matching ones
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)
    return model


# Predict the probabilities for the test data in batches
def predict(model, seq_file, output_file, batch_size=1000, device='cuda'):
    # Read all sequences from the file
    with open(seq_file, 'r') as f:
        sequences = f.readlines()

    total_sequences = len(sequences)
    processed_sequences = 0

    # Open output file for writing predictions
    with open(output_file, 'w') as f_out:
        for start in range(0, total_sequences, batch_size):
            # Get a batch of sequences
            batch = sequences[start:start + batch_size]

            # One-hot encode each sequence
            encoded_batch = [one_hot_encode(seq.strip()) for seq in batch]

            # Ensure the shape is (batch_size, seq_len, input_size)
            encoded_batch = np.array(encoded_batch)
            encoded_batch = torch.tensor(encoded_batch, dtype=torch.float32).to(device)

            # Make predictions with the model
            with torch.no_grad():
                outputs = model(encoded_batch)  # Only pass sequence

            # Compute probabilities using softmax
            softmax = torch.nn.Softmax(dim=1)  # Apply softmax along the second dimension (classes)
            probs = softmax(outputs)  # Get probabilities for each class

            # Write results to the output file
            for seq_idx in range(len(batch)):
                prob_class_1 = round(float(probs[seq_idx][1].item()), 4)
                f_out.write(f"{batch[seq_idx].strip()}\t{prob_class_1}\n")

            # Update the processed count and print progress
            processed_sequences += len(batch)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            memory_usage = psutil.virtual_memory().percent
            memory_usage_Mb = get_memory_usage()
            print(f"{current_time} | Processed {processed_sequences}/{total_sequences} sequences | "
                  f"Memory Usage: {memory_usage_Mb:.2f} MB ({memory_usage}%)")

            # Optionally free memory (if using GPU)
            torch.cuda.empty_cache()


def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in seq])


def main():
    model_path = sys.argv[1]  # '/path/to/model.pth'
    seq_file = sys.argv[2]  # '/path/to/test_sequences.txt'
    output_file = sys.argv[3]  # '/path/to/output_predictions.csv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the trained model
    model = load_model(model_path, device=device)

    # Make predictions and save results
    predict(model, seq_file, output_file, device=device)

if __name__ == "__main__":
    main()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time}: Selecting Finished\n")
