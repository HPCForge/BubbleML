#!/bin/bash

mkdir bubbleml_data && cd bubbleml_data

# Download the Single Bubble Simulation
wget https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/single-bubble.tar.gz 
tar -xvzf single-bubble.tar.gz && rm single-bubble.tar.gz

# Download the Saturated Pool Boiling study consisting of 13 simulations
wget https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-saturated-fc72-2d.tar.gz
tar -xvzf pool-boiling-saturated-fc72-2d.tar.gz && rm pool-boiling-saturated-fc72-2d.tar.gz

# Download the Subcooled Pool Boiling study consisting of 15 simulations
wget https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-subcooled-fc72-2d.tar.gz
tar -xvzf pool-boiling-subcooled-fc72-2d.tar.gz && rm pool-boiling-subcooled-fc72-2d.tar.gz

# Download the Gravity Pool Boiling study consisting of 9 simulations
wget https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-gravity-fc72-2d.tar.gz
tar -xvzf pool-boiling-gravity-fc72-2d.tar.gz && rm https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-gravity-fc72-2d.tar.gz

# Download the Flow Boiling Inlet Velocity study consisting of 8 simulations
wget https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/flow-boiling-velscale-fc72-2d.tar.gz
tar -xvzf flow-boiling-velscale-fc72-2d.tar.gz && rm flow-boiling-velscale-fc72-2d.tar.gz

# Download the Flow Boiling Gravity study consisting of 6 simulations
wget https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/flow-boiling-gravity-fc72-2d.tar.gz
tar -xvzf flow-boiling-gravity-fc72-2d.tar.gz && rm flow-boiling-gravity-fc72-2d.tar.gz

# Download the 3D Earth gravity simulation
# Download the 3D ISS gravity simulation
# Download the 3D Flow Boiling simulation
