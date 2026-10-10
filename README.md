## Workflow

```text
Traffic Matrix Dataset
          |
          v
     Preprocessing
          |
          v
     Traffic Demands
          |
          v
    MILP Optimizer
          |
          | Sequential routing and updates
          v
Optimal Routing Paths and Network State Dataset
          |
          v
  GNN Training with Message Passing
          |
          v
  NNConv: Learning from Node and Edge Features
          |
          v
      Path Reconstruction
          |
          v
   Real-Time Route Prediction
```

## Dataset

The model is trained using the publicly available GEANT network dataset.

- Network topology: 23 nodes, 74 directed edges
- Traffic data: 11,460 Traffic Matrices (TMs)
- Sampling interval: 15 minutes
- Observation period: ~4 months

Traffic matrices provide aggregate network demand rather than individual flow information. Following the methodology of Reis et al. (IJCNN 2019), individual network flows are generated from each traffic matrix.

Each flow is represented as:

f = (source, destination, bandwidth demand, arrival time, duration)

Flow arrival times are shifted by 900 seconds between consecutive traffic matrices to reproduce the sequence of traffic demands that agrees with traffic matrices.

Dataset is available here:
S. Uhlig et al., "Providing Public Intradomain Traffic Matrices to the Research Community", SIGCOMM CCR, 2006.


## MILP-Based Routing Framework

 Built a PuLP-based a Mixed Integer Linear Programming (MILP) solver with cost or optimization function 

 
<img width="292" height="91" alt="image" src="https://github.com/user-attachments/assets/a47361d7-5218-4b72-91c8-8ca7c41bc434" />

 where λ is a small constant edge penalty. The routing problem is therefore formulated as selecting the feasible path with minimum cost, This objective naturally discourages the use of highly utilized edges while simultaneously penalizing unnecessarily long paths. Consequently, incoming flows are preferentially routed through less congested regions of the network, leading to a more balanced utilization of network resources.


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


## Repository Structure

### MILP.py

Implements the Mixed Integer Linear Programming formulation used to generate optimal routing solutions. The solver output serves as the ground-truth labels for training the surrogate machine learning models.

### preprocessing Traffic Matrix.ipynb

Generates traffic demands, preprocesses traffic matrices and prepares graph datasets for routing.

### GNN.py

Implements the Graph Neural Network routing model.

### DNN.py

Implements a fully connected neural network baseline used for comparison against graph-based approaches.


## Project Background

This work was developed as part of the National Supercomputing Mission (NSM) under the MPPLab project at C-DAC.

Preliminary results from this work were presented at:

### CASML 2025

International Conference on Applied AI and Scientific Machine Learning (CASML 2025, IISc)


