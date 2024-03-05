#!/bin/bash

# Specify the input CSV file name
INPUT_CSV="input.csv"
KERNEL_FILE="kernel.txt"

# Check if the input CSV file and KERNEL_FILE exist
if [ ! -f "$INPUT_CSV" ]; then
    echo "Input CSV file '$INPUT_CSV' not found."
    exit 1
fi

if [ ! -f "$KERNEL_FILE" ]; then
    echo "Kernel file '$KERNEL_FILE' not found."
    exit 1
fi

# Create an error log file
ERROR_LOG="errors.log"
> "$ERROR_LOG"

# Read each kernel from the KERNEL_FILE and process each line in the CSV file
while IFS='' read -r KERNEL; do
    while IFS=',' read -r NUM_BLOCKS
    do
        # Run the makefile and build the CUDA program
        make clean
        for f in Power_data*.txt; do
            echo -n "" > "$f"
        done
        if BLOCK_SIZE=$NUM_BLOCKS KERNEL=$KERNEL make BLOCK_SIZE=$NUM_BLOCKS KERNEL=$KERNEL; then
            for i in {1..1}; do
                ./run 
                for f in Power_data*.txt; do
                    cat "$f" >> "../$KERNEL.csv"
                done
            done
        fi
    done < "$INPUT_CSV"
done < "$KERNEL_FILE"
