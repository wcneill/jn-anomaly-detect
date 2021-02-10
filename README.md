# Anomaly Detection.
Research and experimentation with implementing anomaly detection with benchmark datasets and ICEWS data.

## Repo Structure:
Each benchmark dataset gets its own folder `AD_<Benchmark Name>/`. Within each benchmark directory is a set of notebooks. Each notebook contains a different approach to anomaly detection applied to the benchmark defined for that folder. 

The ICEWS dataset is also included, but has it's own README. 

## Python Utility Packages:
The folders `models` and `util` python packages for use in AD. 
- `models/` contains the classes and methods that instantiate and train a variety of types of neural networks used in anomaly detection.

- `util/` primarily contains common data classes, metrics and preprocessing tools. Use of these tools vastly reduces the amount of copied and pasted functions inside the anomaly detection notebooks.


