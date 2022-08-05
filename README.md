# Modeling and Study of Adversarial Attack Arising from Deceiving Perception in Car Autopilot

## [Report](https://drive.google.com/file/d/1oDLT-tGFP5JlN1V_EiNY9McwMdqJnAYp/view) | [Presentation Video](https://drive.google.com/drive/folders/1RHFFfsBU5HvBgxWah8wbz85v0oewqMZr) | [Poster](https://drive.google.com/file/d/1oSgPL7so0WGtu6o-KZZhvHtjLGk7Jyvf/view)

<img src="figures/demo.gif">

This repository contains the code for the project done as a part of the course Trustworthy AI Autonomy at Carnegie Mellon University during Spring 2022 semester by Neel Joshi, Navodit Chandra and Aishwarya Ravi.

### Abstract
Safety of autonomous vehicles has been a matter concern lately. The autopilot system is heavily dependent on AI algorithms, from perception to control. Such systems are vulnerable to adversarial attacks arising from both random as well as intentional practices. This work models a real-life safety-critical scenario, where the moon is perceived as a traffic light, with the help of Digital Twin technology, such that car autopilots can be trained faster using these virtually generated scenarios. We create an end-to-end model starting from sensory perception to generating augmented scenarios based on original scenarios. We also show that majority of such safety-critical scenarios can be easily avoided. For this we demonstrate the efficacy of a real-time solution and a definitive solution.

### Model Pipeline
<img src="figures/pipeline.png">
Figure 1: Framework of adversarial scenario generation; (a) Sensor module, (b) Perception module, (c) Adversarial attack module
