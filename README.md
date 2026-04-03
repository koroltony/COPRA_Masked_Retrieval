# Ultrashort Pulse Phase Retrieval: COPRA Masked Evaluation Testbench

This repository contains a customized simulation suite for ultrashort pulse phase retrieval. Built on top of the open-source **`pypret` (Python Phase Retrieval for Ultrashort Pulses)** library, this project is designed to evaluate the robustness of iterative phase retrieval algorithms (like COPRA) against additive noise. 

It introduces a pipeline to simulate noisy measurement traces, apply various augmentation strategies (masking, spectral subtraction, zero-enforcement), and compare the retrieved pulse against the original pulse using targeted region-of-interest error metrics.

## Foundational Setup
This project extends PyPret's core physics modeling (PNPS) and standard retrieval algorithms. Before modifying this codebase, please review the [official PyPret documentation](https://pypret.readthedocs.io/).

**Prerequisite:** You must generate a simulated dataset of pulses named `pulse_bank.hdf5` using PyPret and place it in the root directory before running these scripts.

---

## Core Features & Methodologies

This framework tests real-world noise mitigation strategies applied to spectrograms (e.g., SHG-FROG, THG-iFROG) prior to algorithm processing:

1. **Peak Energy Masking:** Automatically isolates the highest-energy spectral regions (e.g., the top 10%) and heavily weights them, while blanking out the low-intensity "wings" where noise overpowers the signal.
2. **Dynamic Zero Enforcement:** Analyzes the marginal sums of the raw simulated trace to identify where the signal drops below a 1% threshold, cleaning up background noise in the delay and frequency boundaries.
3. **Spectral Subtraction:** Estimates the DC noise floor from signal-free regions of the trace and subtracts it globally to improve the Signal-to-Noise Ratio (SNR).
4. **Significant-Region MAE:** Standard global Mean Squared Error (MSE) is replaced with targeted Magnitude, Phase, and Complex (Unified) Mean Absolute Error (MAE). These metrics are *only* calculated where the original pulse intensity exceeds 1% of the peak, preventing undefined phase noise in the deep wings from skewing the results.

---

## File Structure & Architecture

### 1. `test_single_retrieval.py` (The Diagnostic Viewer)
Use this script for visual debugging and sanity checks. 
* Pulls a single pulse from the `pulse_bank.hdf5`.
* Runs a side-by-side comparison of a standard noisy retrieval vs. a mitigated/masked retrieval.
* Generates detailed comparative plots highlighting the exact regions where phase error is calculated.

### 2. `mask_error_calculation.py` (The Batch Simulator)
The main script for sweeping parameters.
* Iterates through a defined suite of pulses, measurement schemes, mask sizes, and processing toggles.
* Executes the phase retrieval for every combination and calculates the custom MAE metrics.
* Compiles the massive output dataset into `full_bank_retrieval_results.csv`.

### 3. `mask_error_extraction.py` (The Analytics Pivoter)
A lightweight post-processing script.
* Reads the generated `full_bank_retrieval_results.csv` without rerunning simulations.
* Pivots and aggregates the data to output a clean, ranked terminal summary showing which noise-mitigation configuration performed best for each scheme.

### 4. `benchmarking.py` (The Data Pre-Processor)
Acts as the intermediate wrapper between the simulated physics and the algorithm.
* Injects Gaussian noise into the clean PyPret trace.
* Applies the user-defined background subtractions and dynamic boundaries.
* Constructs the 2D boolean weight arrays (`weights_masked`) that dictate how the retriever handles the data.

### 5. `retriever.py` (The Core Algorithm Logic)
A customized override of PyPret’s `BaseRetriever`.
* **Important Modification:** This file has been altered to natively accept and process 2D `weights` arrays during the iterative minimization loops (e.g., `_error_vector`, `_objective_function`). 

---

## Getting Started

1. **Environment Setup:** Ensure you have Python installed with `numpy`, `pandas`, `matplotlib`, and `pypret`.
2. **Generate Data:** Create your `pulse_bank.hdf5` file containing the test pulses.
3. **Run a Test:** Execute `python test_single_retrieval.py` to verify your environment works and to visually inspect the noise levels on a single FROG/d-scan trace.
4. **Execute a Sweep:** Run `python mask_error_calculation.py`. *(Note: This can be computationally expensive depending on `num_pulses` and `maxiter` limits).*
5. **Analyze:** Run `python mask_error_extraction.py` to view your finalized performance rankings.
