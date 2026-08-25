#!/bin/bash

# Function to validate if input is a number
is_number() {
    [[ $1 =~ ^[0-9]+$ ]]
}

# Function to validate port range
validate_port() {
    if [ $1 -lt 1 ] || [ $1 -gt 65535 ]; then
        return 1
    fi
    return 0
}

# Get port range from user
read -p "Enter starting port number (1-65535): " start_port
read -p "Enter ending port number (1-65535): " end_port

# Validate inputs are numbers
if ! is_number "$start_port" || ! is_number "$end_port"; then
    echo "Error: Please enter valid numbers"
    exit 1
fi

# Validate port range
if ! validate_port "$start_port" || ! validate_port "$end_port"; then
    echo "Error: Ports must be between 1 and 65535"
    exit 1
fi

# Ensure start_port is less than end_port
if [ "$start_port" -gt "$end_port" ]; then
    echo "Error: Starting port must be less than ending port"
    exit 1
fi

echo "Scanning ports $start_port to $end_port..."

# Counter for killed processes
killed_count=0

# Loop through the port range
for port in $(seq "$start_port" "$end_port"); do
    pid=$(lsof -t -i:"$port" 2>/dev/null)
    if [ -n "$pid" ]; then
        process_info=$(ps -p "$pid" -o comm= 2>/dev/null)
        echo "Found process '$process_info' (PID: $pid) on port $port"
        kill -9 "$pid" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "Successfully killed process on port $port"
            ((killed_count++))
        fi
    fi
done

# Summary
echo -e "\nOperation completed"
echo "$killed_count process(es) were terminated"
