#!/bin/bash

input_file="$1"

sum=0
count=0

# Iterate over each line in the file
while read -r line; do
  # Check if the line matches the pattern
  if [[ $line =~ Execution\ Time:\ ([0-9]+\.[0-9]+)\ seconds ]]; then
    # Extract the execution time value from the line
    time="${BASH_REMATCH[1]}"
    
    # Add the execution time to the sum
    sum=$(awk "BEGIN{print $sum + $time}")

    # Increment the count
    ((count++))
  fi
done < "$input_file"

# Calculate the mean
mean=$(awk "BEGIN{print $sum / $count}")

# Print the mean
echo "Mean execution time: $mean seconds"