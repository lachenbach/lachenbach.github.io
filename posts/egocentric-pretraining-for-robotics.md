---
title: Egocentric Pretraining for Robot Foundation Models
description: Assessing the current state.
date: 2026-07-20
draft: true
---

In robot learning today 


## The challenges
1. Human demonstrations are different from robot trajectories in speed and manipulation strategy, which the robot might not be able to perform due to kinematic limitations.
2. If you're training a VLA or similar policy, you'll have to extract actions from human video and account for the resulting embodiment gap.
3. Human hand tracking is often recorded with respect to the camera mounted on the head, which is not static as it is for many bimanual robots.
3. In theory there should be a lot of egocentric data, but in practice there is a limited amount of open source data.
4. 

## Publicly available egocentric datasets
| Name | Size | Task Diversity | Scene Diversity | 
|---|---|---|---|
| EgoVerse-A* | 152h | 3 Single-arm, 3 Bimanual tasks | varies with task |
| EgoVerse-I* | 3192h | >2000 tasks | >240 scenes, >2100 demonstrators |
| EgoDex | 829h | 194 tasks | 


* EgoVerse is split into 6 flagship tasks which make up an EgoVerse-A(cademic) and an EgoVerse-I(industry) split, where the latter has more diverse tasks. The flagship tasks are object-in-container (14.38h), cup-on-saucer (28.90h), bag-grocery (6.78h), fold-clothes (82.96h), scoop-granular (11.13h), sort-utensils (7.67h). The paper reports 75h of human data collected by academic labs, but it seems like industry partners added more flagship data since the release of the paper. Originally, the paper reported 1400h of human data in the I-split which now has grown to around 3200h of data. I did not check the updated task count and diversity but presumably those also grew. 



### Data quality and samples

Below are some samples of egocentric data from EgoVerse. You will notice that there is no standardized understanding of how to collect this data, so even within one dataset, operators and contributors choose different collection strategies.


## Learning policies from egocentric data
| Paper | Human Dataset (h) | Robot Dataset (h) | Dataset Availability | Target Embodiment | Insight |
|---|---:|---:|---|---|---|
| EgoScale | 20.000 | 4 | private | GalaxeaR1Pro, G1 | More human data leads to higher gains for single task post-training |
| S. Kareers Pi Project | 3-5 | ? | private | ARX | Human data is useful if VLA pretraining diversity is scaled |
