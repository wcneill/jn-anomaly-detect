# Anomaly Detection.
Implementation and benchmarking of the Enc-Dec-AD model derived from the paper by Malhotra et. al., found here: https://arxiv.org/pdf/1607.00148.pdf

## Repo Structure:
Each benchmark dataset gets its own folder `AD_<Benchmark Name>/`. Within each benchmark directory is a set of notebooks. Each notebook contains a different approach to anomaly detection applied to the benchmark defined for that folder. 

## Python Utility Packages:
The folders `models` and `util` python packages for use in AD. 
- `models/` contains the classes and methods that instantiate and train a variety of types of neural networks used in anomaly detection.

- `util/` primarily contains common data classes, metrics and preprocessing tools. Use of these tools vastly reduces the amount of copied and pasted functions inside the anomaly detection notebooks.


