## Dataset

The model is trained using the publicly available GEANT network dataset.

- Network topology: 23 nodes, 74 directed edges
- Traffic data: 11,460 Traffic Matrices (TMs)
- Sampling interval: 15 minutes
- Observation period: ~4 months

Traffic matrices provide aggregate network demand rather than individual flow information. Following the methodology of Reis et al. (IJCNN 2019), individual network flows are generated from each traffic matrix.

Each flow is represented as:

f = (source, destination, bandwidth demand, arrival time, duration)

Flow arrival times are shifted by 900 seconds between consecutive traffic matrices to reproduce the temporal evolution of network traffic.

Reference:
S. Uhlig et al., "Providing Public Intradomain Traffic Matrices to the Research Community", SIGCOMM CCR, 2006.


## MILP-Based Routing Framework

Ground-truth routing decisions are generated using a sequential Mixed Integer Linear Programming (MILP) framework.

For each incoming flow:

1. Candidate source-destination paths are generated using NetworkX.
2. Current edge utilizations are maintained dynamically.
3. Capacity constraints are enforced.
4. The path with minimum congestion-aware cost is selected.

The objective function discourages routing through highly utilized links while penalizing unnecessarily long paths.

After every flow arrival or departure:

- Edge utilizations are updated.
- Network state is recorded.
- MILP routing decision is stored.

These network-state and routing-decision pairs form the supervised learning dataset used to train the GNN surrogate.

## Feature Representation

For each routing event, the model receives:

- 74 edge utilization features
- 23 source-node indicators
- 23 destination-node indicators
- 1 normalized bandwidth feature

Total input dimension: 121

Target output:

- 74-dimensional routing vector
- Encodes routing actions on network edges
- Generated directly from MILP solutions

## Graph Neural Network Architecture

The proposed model uses a Message Passing Neural Network based on NNConv layers.

Architecture:

- 3 NNConv message-passing layers
- Hidden dimensions: 46 → 128 → 128 → 128
- Residual connection
- Global mean pooling
- MLP prediction head

The GNN learns routing policies directly from MILP-generated solutions while explicitly exploiting network topology through message passing.

For comparison, a fully-connected DNN baseline is also implemented.


## Project Background

This work was developed as part of the National Supercomputing Mission (NSM) under the MPPLab High Performance Computing and Artificial Intelligence activities at C-DAC.

Preliminary results from this work were presented at:

### CASML 2025

Conference on Advances in Simulation, Machine Learning and Artificial Intelligence (CASML 2025)


