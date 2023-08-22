#!/bin/bash

# Download the Single Bubble Simulation
wget https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/single-bubble.tar.gz 
tar -xvzf single-bubble.tar.gz && rm single-bubble.tar.gz

# Download the Saturated Pool Boiling study consisting of 13 simulations
wget https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-saturated-fc72-2d.tar.gz
tar -xvzf pool-boiling-saturated-fc72-2d.tar.gz && rm pool-boiling-saturated-fc72-2d.tar.gz

# Download the Subcooled Pool Boiling study consisting of 10 simulations
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

# Download the Subcooled Pool Boiling study consisting of 15 simulations with 0.1 temporal resolution
wget https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-subcooled-fc72-2d-0.1.tar.gz
tar -xvzf pool-boiling-subcooled-fc72-2d-0.1.tar.gz && rm pool-boiling-subcooled-fc72-2d-0.1.tar.gz

# Download the Gravity Pool Boiling study consisting of 9 simulations with 0.1 temporal resolution
https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-gravity-fc72-2d-0.1.tar.gz
tar -xvzf pool-boiling-gravity-fc72-2d-0.1.tar.gz && rm pool-boiling-gravity-fc72-2d-0.1.tar.gz

# Download the Flow Boiling Gravity study consisting of 6 simulations with 0.1 temporal resolution
https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/flow-boiling-gravity-fc72-2d-0.1.tar.gz
tar -xvzf flow-boiling-gravity-fc72-2d-0.1.tar.gz && rm flow-boiling-gravity-fc72-2d-0.1.tar.gz

# Download the 3D Pool Boiling Earth gravity simulation
https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-earth-gravity-3d.tar.gz
tar -xvzf pool-boiling-earth-gravity-3d.tar.gz && rm pool-boiling-earth-gravity-3d.tar.gz

# Download the 3D Pool Boiling ISS gravity simulation
https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-iss-gravity-3d.tar.gz
tar -xvzf pool-boiling-iss-gravity-3d.tar.gz && rm pool-boiling-iss-gravity-3d.tar.gz

# Download the 3D Flow Boiling Earth gravity simulation
https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/flow-boiling-earth-gravity-3d.tar.gz
tar -xvzf flow-boiling-earth-gravity-3d.tar.gz && rm flow-boiling-earth-gravity-3d.tar.gz
