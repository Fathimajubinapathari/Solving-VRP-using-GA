# Solving VRP using GA

The attached script is from a project completed during my Master's degree in the Evolutionary Computation course. It focuses on solving the Vehicle Routing Problem (VRP) [1] using a Genetic Algorithm (GA) to find optimal routes for a fleet of trucks, minimizing the total distance traveled. The problem involves determining the best routes for trucks starting from a common station and visiting stops with specific weights, while adhering to load capacities. GA is a stochastic optimization method that mimics the Darwinian theory of survival of the fittest. It uses operators like selection, crossover, and mutation to iteratively improve solution quality over generations [2].

## Input

The input is a CSV file called 'dataset.csv' `dataset.csv`

## Outputs

The script provides the following outputs:

Fitness values for each iteration showing solution improvement.
Best solution details, including truck routes, total CPU time, runtime, number of runs, and relative difference to the optimal solution.

Figures of Vehicle Routes and Fitness Evolution Over Generations.

## Installation

There is a dependency of matplotlib, which can be installed with `pip install -r requirements.txt`. To run the script, use `python vrp.py`. If a run takes too long, consider reducing the `NO_GENERATIONS` or the problem size in general. 

### Note
While developing this script, I referred to similar implementations available for solving related problems as part of my research and learning process.
***

[1] https://en.wikipedia.org/wiki/Vehicle_routing_problem

[2] https://doi.org/10.1007/978-3-319-93025-1_4
