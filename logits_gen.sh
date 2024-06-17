#!/bin/bash

# Set default values for method and router
method="OracleDecoding"
router="normal"

# Define the model size lists
models=("0.5B" "1.8B" "4B" "7B" "14B" "32B" "72B")

# Define the datasets to iterate over
datasets=("gsm8k" "mmlu" "mbpp" "mtbench")

# Set sampling value
sampling=500

# Base paths for models
base_path="Your serial models path"

# Function to compare model sizes without using bc
compare_models() {
    local size1="$1"
    local size2="$2"
    local order=("0.5B" "1.8B" "4B" "7B" "14B" "32B" "72B")
    
    for i in "${!order[@]}"; do
        if [[ "${order[$i]}" == "$size1" ]]; then
            index1=$i
        fi
        if [[ "${order[$i]}" == "$size2" ]]; then
            index2=$i
        fi
    done
    
    if [ $index1 -gt $index2 ]; then
        return 0
    else
        return 1
    fi
}

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    # Iterate over each combination of models
    for large_model in "${models[@]}"; do
        for small_model in "${models[@]}"; do
            # Ensure the large model is greater than the small model
            if compare_models "$large_model" "$small_model"; then
                # Construct the model paths
                if [ "$dataset" == "mtbench" ]; then
                    large_model_path="${base_path}-${large_model}-Chat"
                    small_model_path="${base_path}-${small_model}-Chat"
                else
                    large_model_path="${base_path}-${large_model}"
                    small_model_path="${base_path}-${small_model}"
                fi

                # Print the combination being tested
                echo "Testing with dataset: $dataset, large model: $large_model_path, small model: $small_model_path"

                # Run the Python script with the current combination of parameters
                python logits_gen.py --router $router --method $method --sampling $sampling --dataset $dataset --large-model-path $large_model_path --small-model-path $small_model_path
            fi
        done
    done
done
