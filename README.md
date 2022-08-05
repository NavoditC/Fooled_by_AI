# Modeling and Study of Adversarial Attack Arising from Deceiving Perception in Car Autopilot

## [Report](https://drive.google.com/file/d/1oDLT-tGFP5JlN1V_EiNY9McwMdqJnAYp/view) | [Presentation Video](https://drive.google.com/file/d/1N5V5YexH3902M-3knspH_3AY7XIzYLH6/view?usp=sharing) | [Poster](https://drive.google.com/file/d/1oSgPL7so0WGtu6o-KZZhvHtjLGk7Jyvf/view)

<img src="figures/demo.gif">

This repository contains the code for the project done as a part of the course Trustworthy AI Autonomy at Carnegie Mellon University during Spring 2022 semester by Neel Joshi, Navodit Chandra and Aishwarya Ravi.

### Abstract
Safety of autonomous vehicles has been a matter concern lately. The autopilot system is heavily dependent on AI algorithms, from perception to control. Such systems are vulnerable to adversarial attacks arising from both random as well as intentional practices. This work models a real-life safety-critical scenario, where the moon is perceived as a traffic light, with the help of Digital Twin technology, such that car autopilots can be trained faster using these virtually generated scenarios. We create an end-to-end model starting from sensory perception to generating augmented scenarios based on original scenarios. We also show that majority of such safety-critical scenarios can be easily avoided. For this we demonstrate the efficacy of a real-time solution and a definitive solution.

### Scenario Generation
<img src="figures/pipeline.png">
<p align="center">
  Figure 1: Framework of adversarial scenario generation; (a) Sensor module, (b) Perception module, (c) Adversarial attack module
 </p>

### Generation of Adversarial Images
<img src="figures/pgd_attack.png">
Figure 2: Demonstration of images being perturbed by PGD adversarial attack algorithm

### Image Processing
<img src="figures/image_processing.png">
Figure 3: Steps in processing of adversarial images (a) clean image from sensor (b) simplistic graphics (c) realistic graphics


### Digital Twin
<img src="figures/digital_twin.png">
Figure 4: Left - Scenario from MetaDrive simulator, Right - Same image modified for depicting how the perception algorithm for a real autopilot works

### Results and Analysis
<img src="figures/lunar_cycle.png">
Figure 5: Effect of lunar cycle on probability of autopilot failure. Big corresponds to size of moon when closest to earth and small corresponds to 50% of the full size.


<img src="figures/size_of_moon.png">
Figure 6: Effect of size of moon as perceived from car sensor on autopilot failure


<img src="figures/color_of_moon.png">
Figure 7: Effect of color of moon on autopilot failure

### Adversarial Defense
<img src="figures/randomized_padding.png">
Figure 8: Randomized padding boosts adversarial defense in car autopilot


<img src="figures/adversarial_training.png">
Figure 9: Adversarial training used to make the ResNet-18 neural network architecture robust to PGD attack
