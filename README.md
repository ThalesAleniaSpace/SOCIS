# ESA - Summer of Code in Space (SOCIS)
## [Portable AI solutions for test results analysis - (B12)](https://socis.esa.int/projects/)

##### Scope

The main goal of this project is to train, configure and deploy two deep neural network (a perceptron and a Convolutional Neural Network - (CNN)) in a portable VPU to validate its performances, behavior and integration capacity. The proof of concept will be based on pattern recognition on test results.

---

##### Duration

The project will take place between June and September 2019 and it is divided in 6 sprints of 2 weeks each.

---

##### Sprints

The sprints will be updated when the actual sprint is finished.
* Sprint 1:

  * Technology studying and installing
    * Python 3.7
    * Tensorflow/Keras
    * Jupyter Notebook
    * Docker
  * Input data analysis
    * Analysis of input data
    * Understanding of the data and its categorization
    * Plan how the data will be used
    * Transform the data (if needed) to be used in the network.

* Sprint 2:
  * Implementation of perceptron
    * 4 point solution (ON Graphs)
    * 4 point solution (OFF Graphs)
    * Reduction of points solution (ON graphs)
    * Reduction of points solution (ON graphs)
  * Analyze and optimize the networks
    * Analyze the 2 different solutions
    * Trade off
    * Optimize topology and parameters

* Sprint 3:
  * Create docker image with the perceptron and the REST API
  * Docker image deployed in Thales environment
  * Analyze input data to convolutional network (graphs)
  * Study different convolutional models
  * Trade off of the study made

* Sprint 4
  * First Implementation of CNN
    * Download the CNN model selected in the analysis phase
    * Train the network with a set of images
    * Analyse the time the network takes in training and the training results
  * Install CNN in VPU

* Sprint 5
  * Install CNN in VPU
    * Download a big convolutional model (e.g. RestNet50) and install in the VPU
    * Execute the CNN and compare the results of executing the CNN with or without VPU. Generate a report
    * Test the VPU in MacOS and CentOS  and Windows operative systems

* Sprint 6
  * Write user manual of perceptron
  * Write user manual of CNN
  * Write user manual of VPU

---
