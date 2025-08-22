# run.sh
# !/bin/bash

# --- Configuration ---
INPUT_FILE="inputdata/input_data.root"
MODEL_FILE="models/TrainedGCNModelJun02.pt"
STATS_FILE="stats/normalization_stats.npz"
OUTPUT_FILE="reconstructed_tracks.csv"
TREE_NAME="segs;10" # Adjust this to match the TTree in your new file
ENV_NAME="mu3e_gnn_env"

# --- Script Logic ---

# Check if the conda environment exists
if ! conda env list | grep -q "^\s*$ENV_NAME\s"; then
    echo "Conda environment '$ENV_NAME' not found. Creating it from environment.yml..."
    conda env create -f environment.yml
    if [ $? -ne 0 ]; then
        echo "Failed to create conda environment. Exiting."
        exit 1
    fi
fi

echo "Activating conda environment: $ENV_NAME"
source activate $ENV_NAME

echo "Starting GNN inference pipeline..."

python run_inference.py \
    --input-file "$INPUT_FILE" \
    --tree-name "$TREE_NAME" \
    --model-path "$MODEL_FILE" \
    --stats-path "$STATS_FILE" \
    --output-file "$OUTPUT_FILE" \
    --device "cuda" # or "cpu"

echo "Pipeline finished. Results are in $OUTPUT_FILE"

conda deactivate