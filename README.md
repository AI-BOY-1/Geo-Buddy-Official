# Geo-Buddy: Autonomous Electromagnetic Surveying Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![SimPEG](https://img.shields.io/badge/powered%20by-SimPEG-orange)](https://simpeg.xyz/)

## Overview

This repository contains the reference implementation for the paper:  
**"Adaptive Electromagnetic Surveying via Two-Dimensional Buddy Systems and Game Theory"**

**Geo-Buddy** is an autonomous exploration agent that optimizes geophysical survey designs in real-time. By mapping the "Two-Dimensional Buddy System" (2DBS) from memory management to spatial sampling, and utilizing a Highest Entropy First (HEF) strategy, it achieves order-of-magnitude cost reductions for sparse geological targets.

## Repository Structure

- `core/`: Contains the physics-aware implementation interfacing with **SimPEG** (Simulation and Parameter Estimation in Geophysics).
- `experiments/`: Scripts to reproduce the figures and benchmarks presented in the paper.
- `figures/`: Output directory for generated plots.

## Installation

Ensure you have a Python environment (3.8+) configured. We recommend using Conda.

```bash
# 1. Clone the repository
git clone [https://github.com/AI-BOY-1/Geo-Buddy-Official.git](https://github.com/AI-BOY-1/Geo-Buddy-Official.git)
cd Geo-Buddy

# 2. Install dependencies
pip install -r requirements.txt