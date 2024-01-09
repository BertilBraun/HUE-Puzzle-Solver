# HUE Puzzle Solver

## Overview

This project was developed during one evening in 2017 as a fun exercise to explore different algorithms for solving a unique permutation puzzle. The puzzle consists of an n x n grid where each cell contains a unique number ranging from 0 to n²-1. The objective is to rearrange the cells to achieve a specific target permutation, typically the natural ordering of numbers from 0 to n²-1.

The repository includes four distinct algorithms for solving the puzzle:

1. **Brute Force Algorithm:** Explores all possible states of the puzzle exhaustively to find the solution.
2. **Greedy Algorithm:** At each step, makes the move that seems the best at that moment by maximizing a heuristic function.
3. **Priority Queue (Branch and Bound) Algorithm:** Uses a priority queue to explore states based on a heuristic function, aiming to reduce the number of explored states.
4. **Graph Cycle Detection Algorithm:** Uses a graph cycle detection algorithm to find the solution.

## Puzzle Description

The puzzle is a simplification of the classic hue puzzle where you are given a grid of tiles with different colors and you have to rearrange them to achieve a specific target pattern. The simplification is that in this puzzle, the tiles are numbered from 0 to n²-1 and the target pattern is the natural ordering of numbers from 0 to n²-1. Any two tiles can be swapped directly.

## File Structure

- `solver.py`: This is the main file containing the implementation of the puzzle and the three algorithms for solving it.

## Algorithms

### Brute Force

- **Approach:** Explores all possible states of the puzzle.
- **Usage:** Best for small puzzles due to its exponential time complexity.

### Greedy

- **Approach:** Chooses the best immediate move based on a heuristic.
- **Heuristic Used:** Number of tiles in the correct position.
- **Usage:** Faster than brute force but does not guarantee an optimal solution.

### Priority Queue (Branch and Bound)

- **Approach:** Uses a priority queue to explore states based on their proximity to the solution as estimated by a heuristic.
- **Heuristic Used:** An estimation function that prioritizes nearly solved puzzles.
- **Usage:** More efficient for larger puzzles, balancing between performance and optimality.

### Graph Cycle Detection ([ref](https://www.geeksforgeeks.org/minimum-number-swaps-required-sort-array/))

- **Approach:** Uses a graph cycle detection algorithm to find the solution.
- **Usage:** More efficient for larger puzzles, balancing between performance and optimality.

## Requirements

- Python 3.x
- Numpy and Matplotlib `pip install numpy matplotlib` (in case you want to visualize the puzzle)

## Running the Code

To run the code, simply execute the `solver.py` script in a Python environment:

```bash
python solver.py
```
