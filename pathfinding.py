import numpy as np
import matplotlib.pyplot as plt
import time
from heapq import heappush, heappop
import pandas as pd

# Create a grid with penalty regions
def create_grid(size, penalty_value, penalty_region):
    grid = np.ones((size, size))
    for region in penalty_region:
        x1, y1, x2, y2 = region
        grid[x1:x2+1, y1:y2+1] = penalty_value
    return grid

# Add obstacles to the grid (cells with value 0)
def add_obstacles(grid, obstacle_regions):
    for region in obstacle_regions:
        x1, y1, x2, y2 = region
        grid[x1:x2+1, y1:y2+1] = 0
    return grid

# Generate neighbors, avoiding obstacles
def get_neighbors_no_obstacles(grid, node):
    x, y = node
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] > 0:
            neighbors.append((nx, ny))
    return neighbors

# Euclidean distance heuristic
def euclidean_heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# Reconstruct path from the came_from dictionary
def reconstruct_path(came_from, start, goal, grid):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return None, float('inf')  # No path found
    path.append(start)
    path.reverse()
    cost = sum(grid[node] for node in path)
    return path, cost

# Theta* line-of-sight check
def has_line_of_sight(grid, point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx, sy = (-1 if x1 > x2 else 1), (-1 if y1 > y2 else 1)
    err = dx - dy

    while (x1, y1) != (x2, y2):
        if grid[x1, y1] == 0:
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return grid[x2, y2] > 0

# BFS algorithm
def bfs_no_obstacles(grid, start, goal):
    start_time = time.time()
    queue = [start]
    came_from = {start: None}
    while queue:
        current = queue.pop(0)
        if current == goal:
            break
        for neighbor in get_neighbors_no_obstacles(grid, current):
            if neighbor not in came_from:
                queue.append(neighbor)
                came_from[neighbor] = current
    path, cost = reconstruct_path(came_from, start, goal, grid)
    return path, cost, time.time() - start_time

# DFS algorithm
def dfs_no_obstacles(grid, start, goal):
    start_time = time.time()
    stack = [start]
    came_from = {start: None}
    while stack:
        current = stack.pop()
        if current == goal:
            break
        for neighbor in get_neighbors_no_obstacles(grid, current):
            if neighbor not in came_from:
                stack.append(neighbor)
                came_from[neighbor] = current
    path, cost = reconstruct_path(came_from, start, goal, grid)
    return path, cost, time.time() - start_time

# Dijkstra algorithm
def dijkstra_no_obstacles(grid, start, goal):
    start_time = time.time()
    heap = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while heap:
        current_cost, current = heappop(heap)
        if current == goal:
            break
        for neighbor in get_neighbors_no_obstacles(grid, current):
            new_cost = current_cost + grid[neighbor]
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heappush(heap, (new_cost, neighbor))
                came_from[neighbor] = current
    path, cost = reconstruct_path(came_from, start, goal, grid)
    return path, cost, time.time() - start_time

# A* algorithm
def a_star_no_obstacles(grid, start, goal):
    start_time = time.time()
    heap = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while heap:
        _, current = heappop(heap)
        if current == goal:
            break
        for neighbor in get_neighbors_no_obstacles(grid, current):
            new_cost = cost_so_far[current] + grid[neighbor]
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + euclidean_heuristic(neighbor, goal)
                heappush(heap, (priority, neighbor))
                came_from[neighbor] = current
    path, cost = reconstruct_path(came_from, start, goal, grid)
    return path, cost, time.time() - start_time

# Theta* algorithm
def theta_star(grid, start, goal):
    start_time = time.time()
    open_set = [(0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: euclidean_heuristic(start, goal)}

    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            break

        for neighbor in get_neighbors_no_obstacles(grid, current):
            if came_from[current] and has_line_of_sight(grid, came_from[current], neighbor):
                parent = came_from[current]
                tentative_g_score = g_score[parent] + euclidean_heuristic(parent, neighbor)
            else:
                parent = current
                tentative_g_score = g_score[current] + grid[neighbor]

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = parent
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + euclidean_heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))

    path, cost = reconstruct_path(came_from, start, goal, grid)
    return path, cost, time.time() - start_time

# Visualization function
def visualize_grid(grid, path=None, title=""):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='Greys', origin='upper')
    if path:
        for (x, y) in path:
            plt.plot(y, x, 'ro')
    plt.title(title)
    plt.show()

# Parameters and initial grid setup
grid_size = 20
penalty_value = 6
penalty_regions = [(5, 5, 10, 10)]
obstacle_regions = [(8, 8, 12, 12), (3, 15, 6, 18), (14, 3, 17, 6)]
start_node = (0, 0)
goal_node = (19, 19)

grid = create_grid(grid_size, penalty_value, penalty_regions)
grid_with_obstacles = add_obstacles(grid, obstacle_regions)

# Algorithm definitions
algorithms_with_obstacles = {
    "BFS": bfs_no_obstacles,
    "DFS": dfs_no_obstacles,
    "Dijkstra": dijkstra_no_obstacles,
    "A*": a_star_no_obstacles,
    "Theta*": theta_star,
}

# Run algorithms and visualize results
final_results = {}
for name, algorithm in algorithms_with_obstacles.items():
    path, cost, exec_time = algorithm(grid_with_obstacles, start_node, goal_node)
    final_results[name] = {"path": path, "cost": cost, "time": exec_time}
    visualize_grid(grid_with_obstacles, path, title=f"{name} Path (Cost: {cost}, Time: {exec_time:.4f}s)")

# Display results
final_results_df = pd.DataFrame({
    "Algorithm": list(final_results.keys()),
    "Execution Time (s)": [final_results[a]["time"] for a in final_results],
    "Path Length": [len(final_results[a]["path"]) if final_results[a]["path"] else float('inf') for a in final_results],
    "Path Cost": [final_results[a]["cost"] for a in final_results]
})
print(final_results_df)
