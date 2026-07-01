#!/usr/bin/env python3
"""
Script for extracting cytosine contexts from mitochondrial genomes and running a prediction model.
Usage: python extract_and_predict.py <fasta_file> <ID> <output_directory>
"""

import sys
import os
from pathlib import Path
import subprocess

def read_fasta(fasta_file):
    """Read FASTA file and return a list of (header, sequence) tuples."""
    sequences = []
    with open(fasta_file, 'r') as f:
        current_header = None
        current_sequence = []
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_sequence)))
                current_header = line[1:]  # Remove '>'
                current_sequence = []
            else:
                current_sequence.append(line.upper())  # Ensure uppercase
        
        # Add the last sequence
        if current_header is not None:
            sequences.append((current_header, ''.join(current_sequence)))
    
    return sequences

def extract_c_contexts(sequence, header, context_size=40):
    """
    Extract contexts for all cytosine bases in a sequence.
    Returns a list of (header, position, left_context, c_base, right_context)
    """
    contexts = []
    seq_len = len(sequence)
    
    for i, base in enumerate(sequence):
        if base == 'C':
            pos = i  # 0-based position
            
            # Extract left context (i-40 to i-1)
            left_start = max(0, i - context_size)
            left_context = sequence[left_start:i]
            
            # Pad with Ns if needed
            if len(left_context) < context_size:
                left_context = 'N' * (context_size - len(left_context)) + left_context
            
            # Extract right context (i+1 to i+40)
            right_end = min(seq_len, i + context_size + 1)
            right_context = sequence[i+1:right_end]
            
            # Pad with Ns if needed
            if len(right_context) < context_size:
                right_context = right_context + 'N' * (context_size - len(right_context))
            
            contexts.append((header, pos + 1, left_context, 'C', right_context))  # Convert to 1-based position
    
    return contexts

def main():
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python extract_and_predict.py <fasta_file> <ID> <output_directory>")
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    sample_id = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read FASTA file
    print(f"Reading FASTA file: {fasta_file}")
    sequences = read_fasta(fasta_file)
    
    if not sequences:
        print("Error: No sequences found in the FASTA file.")
        sys.exit(1)
    
    # Step 1: Extract contexts and write to seq4predict.txt
    all_contexts = []
    seq4predict_path = output_path / "seq4predict.txt"
    
    print("Extracting cytosine contexts...")
    with open(seq4predict_path, 'w') as f:
        for header, seq in sequences:
            contexts = extract_c_contexts(seq, header)
            all_contexts.extend(contexts)
            
            # Write 81bp sequences for prediction
            for _, _, left, center, right in contexts:
                combined = left + center + right
                f.write(f"{combined}\n")
    
    print(f"Found {len(all_contexts)} cytosine sites.")
    print(f"Step 1 complete: Contexts written to {seq4predict_path}")
    
    # Step 2: Run the prediction script
    print("\nStep 2: Running prediction model...")
    predict_script = "predict.py"
    model_path = "models/bestmodel.pth"
    input_file = str(seq4predict_path)
    output_file = str(output_path / "predicted_results.txt")
    
    # Check if predict script exists
    if not os.path.exists(predict_script):
        print(f"Error: Prediction script '{predict_script}' not found in current directory.")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)
    
    # Run prediction
    cmd = ["python", predict_script, model_path, input_file, output_file]
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Prediction completed successfully.")
        if result.stdout:
            print(f"Prediction output: {result.stdout[:500]}...")
    except subprocess.CalledProcessError as e:
        print(f"Error running prediction script: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    
    # Step 3: Read prediction results and generate final output
    print("\nStep 3: Generating final output file...")
    pred_results_path = output_path / "predicted_results.txt"
    
    if not os.path.exists(pred_results_path):
        print(f"Error: Prediction results file not found: {pred_results_path}")
        sys.exit(1)
    
    # Read probability values
    probabilities = []
    with open(pred_results_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    probabilities.append(parts[1])  # Second column
                else:
                    print(f"Warning: Unexpected format in prediction results: {line}")
                    probabilities.append("NA")
    
    # Check that we have the same number of predictions as contexts
    if len(probabilities) != len(all_contexts):
        print(f"Warning: Mismatch between number of contexts ({len(all_contexts)}) and predictions ({len(probabilities)})")
        # If fewer predictions, pad with NA
        if len(probabilities) < len(all_contexts):
            probabilities.extend(["NA"] * (len(all_contexts) - len(probabilities)))
        # If more predictions, truncate
        else:
            probabilities = probabilities[:len(all_contexts)]
    
    # Write final output file
    output_file_path = output_path / f"{sample_id}.txt"
    
    with open(output_file_path, 'w') as f:
        # Write header
        f.write("Gene\tPosition\tLeft motifs\tCentral C\tRight motifs\tProbability value\n")
        
        # Write data
        for i, (header, pos, left, center, right) in enumerate(all_contexts):
            prob = probabilities[i] if i < len(probabilities) else "NA"
            f.write(f"{header}\t{pos}\t{left}\t{center}\t{right}\t{prob}\n")
    
    print(f"Step 3 complete: Final results written to {output_file_path}")
    print(f"Total cytosine sites processed: {len(all_contexts)}")

if __name__ == "__main__":
    main()