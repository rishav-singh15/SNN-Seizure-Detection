# Hypergraph SNN EEG Seizure Detection (Self-Built Rebuild)

This repository contains my independent, self-coded rebuild of an SNN-based EEG seizure detection pipeline, built from the ground up using PyTorch, snnTorch, and MNE-Python. All code in this repository was written by me, without LLM assistance, using the original research papers and project specification purely as a reference to check correctness against, not as source to copy from.

This is Prototype 2 of a larger capstone project. An earlier prototype (Prototype 1) established the research direction, literature backbone, and initial experimental results using LLM-assisted code. That prototype is not part of this repository. This repository is a clean rebuild aimed at deepening my own understanding of the tools and pipeline before extending the work further.

## Project Goal

The long term capstone project accelerates a Spiking Neural Network on FPGA hardware for EEG seizure detection. It is not a pure accuracy focused ML project and not a pure hardware efficiency project. Diagnostic accuracy is used as a validation metric, while the core contribution is a hardware mapped hypergraph SNN architecture, where a hypergraph representation is used to decide how a trained SNN's neurons are physically placed onto FPGA cores.

Dataset: CHB-MIT Scalp EEG Database, patient chb01.

## Current Status

This repository currently covers the following completed steps of the rebuild:

- Step A: Data visualization
- Step B: Spike encoding and windowing
- Step C1: Model architecture
- Step C2: Training loop

Steps beyond C2 are in progress and are described in the Roadmap section below.

## Repository Structure

```
.
├── notebooks/
│   ├── step_A_data_visualization.ipynb
│   ├── step_B_encoding_windowing.ipynb
│   ├── step_C1_model_architecture.ipynb
│   └── step_C2_training_loop.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Notebook Descriptions

### Step A: Data Visualization
Loads a single EDF recording (chb01_16) using MNE's `read_raw_edf`. Plots the four target channels (F7-T7, T7-P7, F8-T8, T8-P8) with the seizure interval marked directly on the plot, and visualizes the 16-level one-hot spike encoding for a short time window. This confirms the raw data and encoding scheme are understood correctly before any modeling begins.

### Step B: Encoding and Windowing
Reimplements two pieces of logic independently, from the paper specification rather than from any prior notebook:

- A min/max binning one-hot spike encoder, 16 levels per channel.
- Windowing and labeling logic using 2048 step windows, with a 50 percent overlap rule to decide when a window counts as a seizure label. This is deliberately separated from the encoding step as its own piece of logic.

Output was diffed against a prior reference implementation on chb01_16 to confirm correctness.

### Step C1: Model Architecture
Builds the 64 to 8 to 1 Leaky Integrate and Fire (LIF) network from scratch. Bias is enabled on both layers, and reset by subtraction dynamics are used, matching the recommended mechanism from the project's DNN to SNN conversion reference. The model is verified to instantiate with exactly 529 parameters, matching the target hardware baseline paper. A forward pass on dummy data is run to confirm there are no shape errors, and spike and membrane traces are plotted for inspection.

### Step C2: Training Loop
A standalone script that reloads EEG data, reruns the Step B encoding and windowing logic across multiple chb01 records, defines the Step C1 model, and trains it on a single fold with a small epoch count. This step is about proving the training loop itself is correct, not producing final results. Validation loss is confirmed to decrease visibly over training, and accuracy, confusion matrix, and ROC AUC are computed to sanity check the loop, loss function, and backward pass.

## Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

The notebooks were developed and run on Google Colab. If running locally, update the data loading cells (Google Drive mount and zip extraction) to point to a local copy of the CHB-MIT chb01 data instead.

## Roadmap

The following steps are planned next, continuing directly from Step C2 and tying back into the broader capstone plan.

### Step C3: Vectorized Training and Full Evaluation
Replace the current per-timestep Python loop with snnTorch's vectorized, RNN-style forward pass to remove the sequential bottleneck. Scale up training to the full 7-fold leave-one-record-out evaluation with a realistic epoch count. Target is to match or beat an AUC of approximately 0.998 on this self-built code.

### Step D: Scaled and Sparser Model
Retrain a deliberately larger and sparser version of the SNN so that hardware mapping gains from hypergraph aware partitioning can be demonstrated on a real trained model, rather than only on a synthetic network. This directly closes the main open gap identified in the earlier prototype, where a small, densely connected trained model did not have enough structure to show a benefit from hypergraph partitioning.

### Hypergraph Aware Hardware Mapping
Apply a directed hypergraph representation to the trained SNN, following the reasoning that a standard graph model overcounts hardware signal traffic. In a plain graph, one neuron broadcasting to several receivers in the same physical core is counted as multiple signals, when physically it is a single signal replicated locally. Modeling the SNN as a directed hypergraph instead enables two optimizations:

- Synaptic reuse through partitioning, grouping neurons that share incoming connections into the same physical core.
- Connection locality through placement, positioning heavily communicating cores physically close to each other.

This partitioning and placement work is computed offline and produces a static hardware mapping blueprint. Earlier synthetic network experiments showed a consistent reduction in inter-core message traffic of roughly 24 percent using hypergraph aware partitioning compared to naive partitioning. The goal of Step D and this stage is to reproduce that benefit on the actual trained model.

### FPGA Implementation in Verilog
Translate the trained and hardware mapped SNN into a Verilog implementation targeting FPGA deployment. This includes:

- Implementing the LIF neuron dynamics and quantized weights in synthesizable Verilog.
- Applying the hypergraph derived partitioning and placement blueprint to the physical layout of neurons across FPGA logic resources.
- Validating functional correctness of the Verilog implementation against the trained PyTorch model, using the same evaluation data.
- Measuring hardware level metrics such as resource utilization, inference latency, and estimated energy per inference, and comparing them against the primary hardware baseline this project targets.

This stage completes the pipeline from trained model to hardware mapped structure to a working FPGA implementation, closing the loop between the ML and VLSI halves of the capstone project.

## Notes on Scope

- This repository intentionally excludes Prototype 1. The research direction, literature review, and initial experimental findings from that phase remain valid and inform the roadmap above, but the code here is a separate, self-written implementation.
- All code in this repository was written independently, with the original papers and project specification used only for verification, not as a source to copy from.
