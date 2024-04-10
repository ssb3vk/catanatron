#!/bin/bash

# Loop 10 times
for i in {1..10}
do
    # Generate a timestamp in the format YYYY-MM-DD-HH-MM-SS
    timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
    
    # Create a new directory for this iteration's output
    mkdir -p "data/${timestamp}"
    
    # Run the catanatron-play command with the output directed to the new directory
    catanatron-play --players=AB,AB --num=1000 --output="data/${timestamp}/" --csv
    
    echo "Run $i completed and data saved in data/${timestamp}/"
done