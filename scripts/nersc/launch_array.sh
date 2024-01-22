#!/bin/bash

# Read arguments from a text file
while IFS= read -r line; do
    # Split the line into five sets of arguments
    set -- $line

    # Submit a job using sbatch
    #sbatch your_job_script.sh "$1" "$2"
    echo "$1" "$2"
done < args.txt
