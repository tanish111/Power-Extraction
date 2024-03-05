#!/bin/bash
export LD_LIBRARY_PATH=/home/tanish/Downloads/Power_Prediction/PowerAPI
GREEN='\033[0;32m'
YELLOW='\e[33m'
ENDCOLOR="\e[0m"
# Get the current date and time
formatted_time=$(date +"%Y-%m-%d_%T")
cd template
# Create an error log file
ERROR_LOG="errors_logs_${formatted_time}.log" 
touch $ERROR_LOG
echo "This are error logs for Linear Algebra" > $ERROR_LOG
for folder in */; do
    # Perform actions inside the sub-folder (you can add more commands here)
    folder_name="${folder%/}"
    echo -e "${GREEN}[INFO] Running: $folder_name${ENDCOLOR}"
    cd $folder_name
    # Specify the input CSV file name
    INPUT_CSV="input.csv"
    KERNEL_FILE="kernel.txt"

    # Check if the input CSV file and KERNEL_FILE exist
    if [ ! -f "$INPUT_CSV" ]; then
        echo "Error for $folder:- Input CSV file '$INPUT_CSV' not found." >> $ERROR_LOG
        exit 1
    fi

    if [ ! -f "$KERNEL_FILE" ]; then
        echo "Error for $folder:- Kernel file '$KERNEL_FILE' not found."  >> $ERROR_LOG
        exit 1
    fi

    # Read each kernel from the KERNEL_FILE and process each line in the CSV file
    while IFS='' read -r KERNEL; do
        echo -e "${YELLOW}[INFO] Running Kernel: $KERNEL${ENDCOLOR}"
        while IFS=',' read -r NUM_BLOCKS
        do
            echo "[INFO] Calculating for BlockSize: $NUM_BLOCKS"
            # Run the makefile and build the CUDA program
            make clean > /dev/null 2>&1
            for f in Power_data*.txt; do
                echo -n "" > "$f"
            done
            if BLOCK_SIZE=$NUM_BLOCKS KERNEL=$KERNEL make BLOCK_SIZE=$NUM_BLOCKS KERNEL=$KERNEL > /dev/null 2>&1; then
                for i in {1..1}; do
                    ./run > /dev/null 2>&1
                    for f in Power_data*.txt; do
                        cat "$f" >> "../$KERNEL.csv"
                    done
                done
            fi
        done < "$INPUT_CSV"
    done < "$KERNEL_FILE"
    # Come out of the sub-folder
    cd ..
    echo "${GREEN}Completed Running: $folder_name${ENDCOLOR}"
done