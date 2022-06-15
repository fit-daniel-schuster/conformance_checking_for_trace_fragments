### Overview
This repository contains the prototype implementation of the proposed infix-/postfix-alignment approach presented in the paper 
***Conformance Checking for Trace Fragments Using Infix- and Postfix-Alignments*** 
by Daniel Schuster, Niklas Föcking, Sebastiaan J. van Zelst, and Wil M. P. van der Aalst.

Corresponding author: Daniel Schuster ([Mail](mailto:daniel.schuster@fit.fraunhofer.de?subject=github-incremental_a_star_approach))


### Repository Structure
* The two proposed algorithms are implemented in 
`infix_alignments/alignments/infix_alignments/variants/baseline_approach.py` and `infix_alignments/alignments/infix_alignments/variants/tree_based_preprocessing.py`.
* In `experiments/infix_alignments/experiments.py` is an example script how to run the conducted experiments.

### Experiments
We provide a Dockerfile to run the conducted experiments. The following commands execute the experiments for all event logs in the provided directory (`<dir>`) and a noise threshold of the Inductive Miner of 0.9.
```shell
docker build -t infix-alignment-experiments .
docker run -it -e MODE="dir" -e NOISE_THRESHOLD="0.9" -v <dir>:/usr/data infix-alignment-experiments
```

### Further Results

This repository contains additional results not presented in the paper.
The results/plots are located in `experimental_results/`.
We use four different process models per event log, e.g., the directory `IMf_0.9` contains the results for all tested event logs when using the model returned by the Inductive Miner (IMf) with threshold of 0.9.