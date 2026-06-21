"""
Deep Observation Maze Solver – with GIF Export
Records the entire solving sequence and exports it as an animated GIF.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import heapq
import random
import time
from PIL import Image, ImageGrab
import imageio.v2 as imageio
import os

# ------------------------------------------------------------------------------
# Maze Generator
# ------------------------------------------------------------------------------

class MazeGenerator:
    def __init__(self, width=21, height=21):
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.grid = None
        self.start = (1, 1)
        self.goal = (self.width - 2, self.height - 2)

    def generate(self):
        self.grid = np.zeros((self.height, self.width), dtype=int)
        stack = [(1, 1)]
        self.grid[1, 1] = 1
        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(2,0), (-2,0), (0,2), (0,-2)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] == 0:
                    neighbors.append((nx, ny, dx, dy))
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                self.grid[y + dy//2, x + dx//2] = 1
                self.grid[ny, nx] = 1
                stack.append((nx, ny))
            else:
                stack.pop()
        self.grid[self.start[1], self.start[0]] = 1
        self.grid[self.goal[1], self.goal[0]] = 1

    def is_free(self, x, y):
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        return self.grid[y, x] == 1

    def get_free_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if self.is_free(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

# ------------------------------------------------------------------------------
# HierarchicalPathfinder
# ------------------------------------------------------------------------------

class HierarchicalPathfinder:
    def __init__(self, maze, agent_pos=None, num_layers=3):
        self.maze = maze
        if agent_pos is None:
            agent_pos = maze.start
        self.agent_pos = agent_pos
        self.num_layers = num_layers
        self.radii = [8, 4, 2]
        self.noise_vars = [0.2, 0.1, 0.05]
        self.belief_maps = [np.ones((maze.height, maze.width)) * 0.5 for _ in range(num_layers)]
        self.true_occupancy = maze.grid.copy()
        self.visited = set()
        self.visited.add(agent_pos)

    def observe(self):
        x, y = self.agent_pos
        observations = []
        for layer_idx in range(self.num_layers):
            radius = self.radii[layer_idx]
            noise_var = self.noise_vars[layer_idx]
            obs_map = np.ones((self.maze.height, self.maze.width)) * 0.5
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < self.maze.width and 0 <= ny < self.maze.height:
                        true_occ = self.true_occupancy[ny, nx]
                        true_prob = 1.0 if true_occ == 1 else 0.0
                        obs = true_prob + np.random.normal(0, np.sqrt(noise_var))
                        obs = np.clip(obs, 0.0, 1.0)
                        obs_map[ny, nx] = obs
            observations.append(obs_map)
        return observations

    def upward_pass(self, observations):
        for i in range(self.num_layers):
            if i == 0:
                prior_var = self.noise_vars[i] * 0.5
                obs_var = self.noise_vars[i]
                posterior = (prior_var * observations[i] + obs_var * self.belief_maps[i]) / (prior_var + obs_var)
                self.belief_maps[i] = posterior
            else:
                prior = self.belief_maps[i-1]
                obs = observations[i]
                prior_var = self.noise_vars[i-1] * 0.5
                obs_var = self.noise_vars[i]
                posterior = (prior_var * obs + obs_var * prior) / (prior_var + obs_var)
                self.belief_maps[i] = posterior

    def downward_pass(self):
        for i in range(self.num_layers-2, -1, -1):
            prior = self.belief_maps[i+1]
            current = self.belief_maps[i]
            alpha = 0.3
            self.belief_maps[i] = alpha * prior + (1 - alpha) * current

    def step(self, iterations=1):
        obs = self.observe()
        for _ in range(iterations):
            self.upward_pass(obs)
            self.downward_pass()

    def get_fused_map(self):
        return self.belief_maps[0]

    def get_uncertainty_map(self):
        fused = self.get_fused_map()
        return fused * (1 - fused)

    def plan_path(self, start=None, goal=None, threshold=0.45):
        if start is None:
            start = self.agent_pos
        if goal is None:
            goal = self.maze.goal
        fused = self.get_fused_map()
        obstacle_map = (fused < threshold).astype(int)
        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        visited_set = set()
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            if current in visited_set:
                continue
            visited_set.add(current)
            x, y = current
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.maze.width and 0 <= ny < self.maze.height:
                    if obstacle_map[ny, nx] == 1:
                        continue
                    neighbor = (nx, ny)
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def move_agent(self, new_pos):
        self.agent_pos = new_pos
        self.visited.add(new_pos)

    def explore(self):
        x, y = self.agent_pos
        neighbors = self.maze.get_free_neighbors(x, y)
        if not neighbors:
            return False
        uncertainty = self.get_uncertainty_map()
        unvisited = [n for n in neighbors if n not in self.visited]
        if unvisited:
            candidates = unvisited
        else:
            candidates = neighbors
        best = max(candidates, key=lambda p: uncertainty[p[1], p[0]] + random.uniform(-0.01, 0.01))
        self.agent_pos = best
        self.visited.add(best)
        return True

# ------------------------------------------------------------------------------
# GUI Application
# ------------------------------------------------------------------------------

class MazeSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Observation Maze Solver – with GIF Export")
        self.maze = MazeGenerator(21, 21)
        self.maze.generate()
        self.agent_pos = self.maze.start
        self.goal = self.maze.goal
        self.pathfinder = HierarchicalPathfinder(self.maze, self.agent_pos, num_layers=3)
        self.cell_size = 20
        self.canvas_width = self.maze.width * self.cell_size
        self.canvas_height = self.maze.height * self.cell_size
        self.steps = 0
        self.create_widgets()
        self.update_display()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        canvas_frame = ttk.LabelFrame(main_frame, text="Maze & Belief", padding=5)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, bg='#1e2a3a')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=5)
        control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        ttk.Label(control_frame, text="Move Agent:").grid(row=0, column=0, columnspan=2)
        move_buttons = ttk.Frame(control_frame)
        move_buttons.grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Button(move_buttons, text="↑", command=lambda: self.move(0,-1)).grid(row=0, column=1)
        ttk.Button(move_buttons, text="←", command=lambda: self.move(-1,0)).grid(row=1, column=0)
        ttk.Button(move_buttons, text="→", command=lambda: self.move(1,0)).grid(row=1, column=2)
        ttk.Button(move_buttons, text="↓", command=lambda: self.move(0,1)).grid(row=2, column=1)

        ttk.Button(control_frame, text="Observe & Update", command=self.observe_step).grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Plan Path (A*)", command=self.plan_path).grid(row=3, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Step Solve", command=self.step_solve).grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Auto Solve", command=self.auto_solve).grid(row=5, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Export GIF", command=self.export_gif).grid(row=6, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Reset", command=self.reset).grid(row=7, column=0, columnspan=2, pady=5)

        info_frame = ttk.LabelFrame(control_frame, text="Status", padding=5)
        info_frame.grid(row=8, column=0, columnspan=2, sticky="nsew", pady=10)
        self.info_text = tk.Text(info_frame, height=12, width=30, bg='#0a0f1a', fg='#b0c6ff', font=('Courier', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def move(self, dx, dy):
        x, y = self.agent_pos
        nx, ny = x+dx, y+dy
        if self.maze.is_free(nx, ny):
            self.agent_pos = (nx, ny)
            self.pathfinder.move_agent(self.agent_pos)
            self.update_display()

    def observe_step(self):
        self.pathfinder.step(iterations=1)
        self.update_display()

    def plan_path(self):
        path = self.pathfinder.plan_path(start=self.agent_pos, goal=self.goal, threshold=0.45)
        if path is None:
            messagebox.showinfo("Path", "No path found with current belief!")
        else:
            self.draw_path(path, color='white')
            self.info_text.insert(tk.END, f"\nA* path length: {len(path)} steps.")
            self.info_text.see(tk.END)

    def step_solve(self):
        path = self.pathfinder.plan_path(start=self.agent_pos, goal=self.goal, threshold=0.45)
        if path is not None and len(path) >= 2:
            next_pos = path[1]
            self.agent_pos = next_pos
            self.pathfinder.move_agent(next_pos)
            self.pathfinder.step(iterations=1)
            self.steps += 1
        else:
            moved = self.pathfinder.explore()
            if moved:
                self.agent_pos = self.pathfinder.agent_pos
                self.pathfinder.step(iterations=1)
                self.steps += 1
                self.info_text.insert(tk.END, f"Explored to {self.agent_pos}\n")
            else:
                self.info_text.insert(tk.END, "Stuck: no free neighbor!\n")
        self.update_display()
        if self.agent_pos == self.goal:
            self.info_text.insert(tk.END, f"🎉 Goal reached in {self.steps} steps!\n")
        else:
            self.info_text.insert(tk.END, f"Step {self.steps}: at {self.agent_pos}\n")
        self.info_text.see(tk.END)

    def auto_solve(self):
        self.info_text.insert(tk.END, "\nAuto-solving with exploration...\n")
        self.info_text.see(tk.END)
        while self.agent_pos != self.goal:
            path = self.pathfinder.plan_path(start=self.agent_pos, goal=self.goal, threshold=0.45)
            if path is not None and len(path) >= 2:
                next_pos = path[1]
                self.agent_pos = next_pos
                self.pathfinder.move_agent(next_pos)
                self.pathfinder.step(iterations=1)
                self.steps += 1
            else:
                moved = self.pathfinder.explore()
                if moved:
                    self.agent_pos = self.pathfinder.agent_pos
                    self.pathfinder.step(iterations=1)
                    self.steps += 1
                else:
                    self.info_text.insert(tk.END, "Stuck! Cannot move.\n")
                    break
            self.update_display()
            self.root.update()
            time.sleep(0.05)
        if self.agent_pos == self.goal:
            self.info_text.insert(tk.END, f"✅ Goal reached in {self.steps} steps!\n")
        else:
            self.info_text.insert(tk.END, "Failed to reach goal.\n")
        self.info_text.see(tk.END)

    def export_gif(self):
        """Run auto_solve and record each frame to a GIF."""
        # Ensure we start from a fresh maze
        self.reset()
        # Record frames
        frames = []
        # We'll simulate stepping and capture canvas after each move
        while self.agent_pos != self.goal:
            path = self.pathfinder.plan_path(start=self.agent_pos, goal=self.goal, threshold=0.45)
            if path is not None and len(path) >= 2:
                next_pos = path[1]
                self.agent_pos = next_pos
                self.pathfinder.move_agent(next_pos)
                self.pathfinder.step(iterations=1)
                self.steps += 1
            else:
                moved = self.pathfinder.explore()
                if moved:
                    self.agent_pos = self.pathfinder.agent_pos
                    self.pathfinder.step(iterations=1)
                    self.steps += 1
                else:
                    break
            # Update canvas
            self.update_display()
            self.root.update()
            # Capture canvas as image
            x = self.canvas.winfo_rootx()
            y = self.canvas.winfo_rooty()
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            # Grab the canvas area
            img = ImageGrab.grab(bbox=(x, y, x+w, y+h))
            frames.append(img)
            # Small delay for visibility
            time.sleep(0.05)
        # Save GIF
        if frames:
            gif_path = "maze_solve.gif"
            imageio.mimsave(gif_path, frames, format='GIF', duration=0.2)
            self.info_text.insert(tk.END, f"\nGIF saved as {gif_path}\n")
        else:
            self.info_text.insert(tk.END, "\nNo frames captured.\n")
        self.info_text.see(tk.END)

    def reset(self):
        self.maze = MazeGenerator(21, 21)
        self.maze.generate()
        self.agent_pos = self.maze.start
        self.goal = self.maze.goal
        self.pathfinder = HierarchicalPathfinder(self.maze, self.agent_pos, num_layers=3)
        self.steps = 0
        self.update_display()
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "Reset. New maze.\n")

    def draw_grid(self, map_data, offset=(0,0), cell_size=None):
        if cell_size is None:
            cell_size = self.cell_size
        ox, oy = offset
        h, w = map_data.shape
        for y in range(h):
            for x in range(w):
                val = map_data[y, x]
                if val < 0.4:
                    color = '#cc3333'
                elif val > 0.6:
                    color = '#33cc33'
                else:
                    color = '#666688'
                self.canvas.create_rectangle(ox + x*cell_size, oy + y*cell_size,
                                             ox + (x+1)*cell_size, oy + (y+1)*cell_size,
                                             fill=color, outline='')

    def draw_agent(self):
        x, y = self.agent_pos
        cx = x*self.cell_size + self.cell_size//2
        cy = y*self.cell_size + self.cell_size//2
        r = self.cell_size//3
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill='yellow', outline='white', width=2)

    def draw_goal(self):
        x, y = self.goal
        cx = x*self.cell_size + self.cell_size//2
        cy = y*self.cell_size + self.cell_size//2
        r = self.cell_size//3
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill='cyan', outline='white', width=2)

    def draw_path(self, path, color='white'):
        for px, py in path:
            cx = px*self.cell_size + self.cell_size//2
            cy = py*self.cell_size + self.cell_size//2
            self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill=color, outline=color)

    def update_display(self):
        self.canvas.delete("all")
        fused = self.pathfinder.get_fused_map()
        self.draw_grid(fused, (0,0), self.cell_size)
        self.draw_agent()
        self.draw_goal()
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"Agent at: {self.agent_pos}\nGoal at: {self.goal}\n")
        self.info_text.insert(tk.END, f"Steps: {self.steps}\n")
        self.info_text.see(tk.END)

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    root = tk.Tk()
    app = MazeSolverGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()