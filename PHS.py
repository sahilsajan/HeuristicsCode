import numpy as np
import heapq
import time
import matplotlib.pyplot as plt

# Generate a 50x50 binary matrix
np.random.seed(42)  # For reproducibility
size = 500
path_prob = 0.50
matrix = np.random.choice([0, 1], size=(size, size), p=[path_prob, 1 - path_prob])

# Ensure a guaranteed path exists
matrix[0, :] = 1  # 0th row is walkable
matrix[:, size - 1] = 1  # Last column is walkable

# Define Euclidean distance heuristic
def euclidean_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

# Get valid neighbors of a node
def get_neighbors(matrix, node):
    x, y = node
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size and matrix[nx, ny] == 1:
            neighbors.append((nx, ny))  # Add valid neighbor
    return neighbors

# A* Algorithm with a visited set
def a_star_with_visited(matrix, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    visited = set()  # Set to track visited nodes
    g_score = {start: 0}
    f_score = {start: euclidean_distance(start, goal)}
    nodes_explored = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue  # Skip already processed nodes

        visited.add(current)  # Mark the node as visited
        nodes_explored += 1

        if current == goal:  # If we reach the goal
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()  # Reverse the path to start->goal order
            return path, len(path), nodes_explored

        for neighbor in get_neighbors(matrix, current):
            if neighbor not in visited:  # Only process unvisited neighbors
                tentative_g_score = g_score[current] + 1  # Distance to neighbor
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path and costs for this neighbor
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + euclidean_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))  # Add neighbor to open set

    return [], 0, nodes_explored  # Return empty path if goal is unreachable

# Greedy Best-First Search with visited set
def greedy_best_first_search(matrix, start, goal):
    open_set = []
    heapq.heappush(open_set, (euclidean_distance(start, goal), start))  # Add start node
    came_from = {}
    visited = set()  # Set to track visited nodes
    nodes_explored = 0  # Counter for number of nodes explored

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue  # Skip if the node has already been visited

        visited.add(current)  # Mark the node as visited
        nodes_explored += 1

        if current == goal:  # If we reach the goal
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()  # Reverse the path to start->goal order
            return path, len(path), nodes_explored

        # Explore the neighbors of the current node
        for neighbor in get_neighbors(matrix, current):
            if neighbor not in visited:  # Only process unvisited neighbors
                came_from[neighbor] = current  # Update path
                heapq.heappush(open_set, (euclidean_distance(neighbor, goal), neighbor))  # Add neighbor to open set

    return [], 0, nodes_explored  # Return empty path if goal is unreachable

# Corrected Jump Point Search Implementation
def jump_point_search(matrix, start, goal):
    def jump(x, y, dx, dy):
        # Jump in the given direction until hitting an obstacle or finding a critical point
        while 0 <= x < size and 0 <= y < size and matrix[x, y] == 1:
            # Check if we reached the goal
            if (x, y) == goal:
                return (x, y)
            # Check for forced neighbors
            if dx != 0 and dy != 0:  # Diagonal
                if (0 <= x - dx < size and matrix[x - dx, y] == 0 and 0 <= y + dy < size and matrix[x, y + dy] == 1) or \
                   (0 <= y - dy < size and matrix[x, y - dy] == 0 and 0 <= x + dx < size and matrix[x + dx, y] == 1):
                    return (x, y)
            elif dx != 0:  # Horizontal
                if (0 <= y + 1 < size and matrix[x, y + 1] == 1 and 0 <= x - dx < size and matrix[x - dx, y + 1] == 0) or \
                   (0 <= y - 1 < size and matrix[x, y - 1] == 1 and 0 <= x - dx < size and matrix[x - dx, y - 1] == 0):
                    return (x, y)
            elif dy != 0:  # Vertical
                if (0 <= x + 1 < size and matrix[x + 1, y] == 1 and 0 <= y - dy < size and matrix[x + 1, y - dy] == 0) or \
                   (0 <= x - 1 < size and matrix[x - 1, y] == 1 and 0 <= y - dy < size and matrix[x - 1, y - dy] == 0):
                    return (x, y)

            # Continue jumping
            x += dx
            y += dy
        return None

    def identify_successors(current):
        # Get valid successors by jumping
        successors = []
        x, y = current
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Directions
            next_jump = jump(x + dx, y + dy, dx, dy)
            if next_jump:
                successors.append(next_jump)
        return successors

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: euclidean_distance(start, goal)}
    nodes_explored = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_explored += 1

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, len(path), nodes_explored

        for neighbor in identify_successors(current):
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + euclidean_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return [], 0, nodes_explored

# Start and goal positions
start = (0, 0)  # Top-left corner
goal = (size - 1, size - 1)  # Bottom-right corner

# Measure A* performance
start_time = time.time()
a_star_path, a_star_path_len, a_star_nodes = a_star_with_visited(matrix, start, goal)
a_star_time = time.time() - start_time

# Measure Greedy Best-First Search performance
start_time = time.time()
gbfs_path, gbfs_path_len, gbfs_nodes = greedy_best_first_search(matrix, start, goal)
gbfs_time = time.time() - start_time

# Measure Jump Point Search performance
start_time = time.time()
jps_path, jps_path_len, jps_nodes = jump_point_search(matrix, start, goal)
jps_time = time.time() - start_time

# Collect metrics for analysis
algorithms = ["A*", "Greedy Best-First", "Jump Point Search"]
execution_times = [a_star_time, gbfs_time, jps_time]
path_lengths = [a_star_path_len, gbfs_path_len, jps_path_len]
nodes_explored = [a_star_nodes, gbfs_nodes, jps_nodes]

# Plot Execution Time
plt.bar(algorithms, execution_times, color='blue')
plt.xlabel('Algorithm')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison')
plt.show()

# Plot Path Length
plt.bar(algorithms, path_lengths, color='green')
plt.xlabel('Algorithm')
plt.ylabel('Path Length')
plt.title('Path Length Comparison')
plt.show()

# Plot Nodes Explored
plt.bar(algorithms, nodes_explored, color='orange')
plt.xlabel('Algorithm')
plt.ylabel('Nodes Explored')
plt.title('Nodes Explored Comparison')
plt.show()

# Output results for reference
print("Execution Times:", execution_times)
print("Path Lengths:", path_lengths)
print("Nodes Explored:", nodes_explored)