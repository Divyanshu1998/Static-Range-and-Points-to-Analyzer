#!/usr/bin/env bash

# Install lit if not already installed
lit --version >/dev/null || pip3 install lit psutil --break-system-packages

# Build the analysis
./build.sh || { echo "Building analysis failed. Exiting."; exit; }

# Run lit on the testcase directory
if [ ! -d "testcases" ]
then
    echo "testcases directory not found. Make sure you are running this from the same folder as the script"
    exit
fi
lit testcases -v
