import time
import heapq
import random
from typing import List


PUZZLE_SIZE = (10, 12)
PUZZLE_CELL_COUNT = PUZZLE_SIZE[0] * PUZZLE_SIZE[1]

def time_me(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken by '{func.__name__}': {end_time - start_time} seconds")
        return result
    return wrapper

class PuzzleState:
    def __init__(self, tiles, swaps=0):
        self.tiles = tuple(tiles)  # The tiles are stored in a flat list.
        self.swaps = swaps

    def is_solved(self) -> bool:
        """ Check if the puzzle is solved """
        return self.tiles == tuple(range(PUZZLE_CELL_COUNT))

    def generate_next_states(self) -> List["PuzzleState"]:
        """ Generate all possible next states from this state """
        next_states = []
        for i in range(len(self.tiles)):
            for j in range(i + 1, len(self.tiles)):
                new_tiles = list(self.tiles)
                new_tiles[i], new_tiles[j] = new_tiles[j], new_tiles[i]
                next_states.append(PuzzleState(new_tiles, self.swaps + 1))
        return next_states

    def heuristic(self) -> int:
        """ Calculate the heuristic value of the state """
        # for correctness probable required: + self.swaps
        return sum(tile != goal_tile for tile, goal_tile in enumerate(self.tiles)) 
    
    def plot_state(self) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        
        puzzle_grid = np.array(self.tiles).reshape(PUZZLE_SIZE)
        correct_positions = np.arange(PUZZLE_CELL_COUNT).reshape(PUZZLE_SIZE)

        fig, ax = plt.subplots()
        for (j, i), label in np.ndenumerate(puzzle_grid):
            correct_position = np.where(correct_positions == label)
            color = plt.cm.viridis((correct_position[0][0] * PUZZLE_SIZE[1] + correct_position[1][0]) / (PUZZLE_CELL_COUNT - 1))
            # ax.text(i, j, str(label), ha='center', va='center', color='white')
            ax.fill([i, i, i+1, i+1], [j, j+1, j+1, j], color=color)

        plt.yticks(np.arange(PUZZLE_SIZE[0] + 1), [])
        plt.xticks(np.arange(PUZZLE_SIZE[1] + 1), [])
        ax.grid(which="both")
        plt.gca().invert_yaxis()
        plt.show()

    def __repr__(self) -> str:
        return f"State: {self.tiles} ({self.swaps} swaps)"

    def __lt__(self, other) -> bool:
        return self.heuristic() < other.heuristic()

def solve_puzzle(initial_state: PuzzleState) -> int:
    return graph_search(initial_state)
    return priority_search(initial_state)
    return greedy_solve(initial_state)
    return brute_force(initial_state)
    
@time_me
def graph_search(initial_state: PuzzleState) -> int:
    """ Implement a graph search algorithm """
    
    array_pos = [*enumerate(initial_state.tiles)]
    array_pos.sort(key=lambda x: x[1])
    
    visited = set()
    ans = 0
    
    for i in range(len(array_pos)):
        if i in visited or array_pos[i][0] == i:
            continue
        
        cycle_size = 0
        j = i
        
        while not j in visited:
            visited.add(j)
            j = array_pos[j][0]
            cycle_size += 1
            
        if cycle_size > 0:
            ans += (cycle_size - 1)
            
    return ans
    
@time_me
def priority_search(initial_state: PuzzleState) -> int:
    """ Implement a priority search algorithm """
    pq = []
    heapq.heappush(pq, (initial_state.heuristic(), initial_state))

    while pq:
        _, current_state = heapq.heappop(pq)

        if current_state.is_solved():
            return current_state.swaps

        for next_state in current_state.generate_next_states():
            heapq.heappush(pq, (next_state.heuristic(), next_state))

    return None  # In case no solution is found

@time_me
def brute_force(initial_state: PuzzleState) -> int:
    """ Implement a brute force algorithm """
    visited = set()
    solved_states = set()
    queue = [initial_state]
    
    while queue:
        current_state = queue.pop(0)
        
        if current_state.is_solved():
            solved_states.add(current_state)
            
        for next_state in current_state.generate_next_states():
            if next_state.tiles not in visited:
                visited.add(next_state.tiles)
                queue.append(next_state)
            
    return min(solved_states, key=lambda state: state.swaps)

@time_me
def greedy_solve(initial_state: PuzzleState) -> int:
    current_state = initial_state

    while not current_state.is_solved():
        next_states = current_state.generate_next_states()
        current_state = min(next_states, key=lambda state: state.heuristic())

    return current_state


if __name__ == "__main__":
    # Example usage
    puzzle = list(range(PUZZLE_CELL_COUNT))
    random.seed(42)
    random.shuffle(puzzle)

    initial_state = PuzzleState(puzzle)

    swaps = solve_puzzle(initial_state)

    print(f"Solved in {swaps} swaps")
    initial_state.plot_state()
    
    solved_state = PuzzleState(list(range(PUZZLE_CELL_COUNT)), swaps)
    solved_state.plot_state()

