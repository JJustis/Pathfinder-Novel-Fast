"""
Deep Observation Maze Solver – Hierarchical Bayesian Navigation
With GIF recording capability.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import heapq
import random
import time
import sys

# Optional GIF support
try:
    import imageio
    from PIL import Image, ImageDraw
    GIF_AVAILABLE = True
except ImportError:
    GIF_AVAILABLE = False

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
VIEWPORT_CELLS = 41
VIEWPORT_SIZE_PX = 700
MAZE_SIZE = 41

# ------------------------------------------------------------------------------
# 1. MAZE GENERATOR
# ------------------------------------------------------------------------------
class MazeGenerator:
    def __init__(self, width=MAZE_SIZE, height=MAZE_SIZE):
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
# 2. HIERARCHICAL PATHFINDER (core algorithm)
# ------------------------------------------------------------------------------
class HierarchicalPathfinder:
    def __init__(self, world, agent_pos=None, num_layers=3,
                 novelty_weight=1.0, revisit_penalty=2.0):
        self.world = world
        if agent_pos is None:
            agent_pos = world.start if hasattr(world, 'start') else (0,0)
        self.agent_pos = agent_pos
        self.num_layers = num_layers
        self.radii = [8, 4, 2]
        self.noise_vars = [0.2, 0.1, 0.05]
        self.belief_maps = [np.ones((world.height, world.width)) * 0.5
                            for _ in range(num_layers)]
        self.visit_counts = np.zeros((world.height, world.width), dtype=int)
        self.visited = set()
        self.visited.add(agent_pos)
        self.recent_positions = []
        self.failure_count = 0
        self.novelty_weight = novelty_weight
        self.revisit_penalty = revisit_penalty
        self._increment_visit(agent_pos)

    def _increment_visit(self, pos):
        x, y = pos
        if 0 <= x < self.world.width and 0 <= y < self.world.height:
            self.visit_counts[y, x] += 1

    def get_visit_count(self, pos):
        x, y = pos
        if 0 <= x < self.world.width and 0 <= y < self.world.height:
            return self.visit_counts[y, x]
        return 0

    def observe(self):
        ax, ay = self.agent_pos
        observations = []
        for layer_idx in range(self.num_layers):
            radius = self.radii[layer_idx]
            noise_var = self.noise_vars[layer_idx]
            obs_map = np.ones((self.world.height, self.world.width)) * 0.5
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    nx, ny = ax+dx, ay+dy
                    if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                        true_occ = 1 if self.world.is_free(nx, ny) else 0
                        true_prob = 1.0 if true_occ else 0.0
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
            alpha = 0.3
            self.belief_maps[i] = alpha * self.belief_maps[i+1] + (1 - alpha) * self.belief_maps[i]

    def step(self, iterations=1):
        obs = self.observe()
        for _ in range(iterations):
            self.upward_pass(obs)
            self.downward_pass()

    def get_fused_map(self):
        return self.belief_maps[0]

    def get_uncertainty(self, x, y):
        val = self.belief_maps[0][y, x]
        return val * (1 - val)

    def is_cell_free_belief(self, x, y, threshold=0.45):
        if 0 <= x < self.world.width and 0 <= y < self.world.height:
            val = self.belief_maps[0][y, x]
        else:
            return False
        return val >= threshold

    def plan_path(self, start=None, goal=None, threshold=0.45, max_iter=20000):
        if start is None:
            start = self.agent_pos
        if goal is None:
            goal = self.world.goal

        def is_free_belief(x, y):
            return self.is_cell_free_belief(x, y, threshold)

        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        def edge_cost(current, neighbor):
            visits = self.get_visit_count(neighbor)
            cost = 1.0 + self.revisit_penalty * visits - self.novelty_weight * (1.0 / (1.0 + visits))
            return max(0.1, cost)

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: heuristic(start, goal)}
        closed = set()
        iter_count = 0

        while open_set and iter_count < max_iter:
            current = heapq.heappop(open_set)[1]
            iter_count += 1
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            if current in closed:
                continue
            closed.add(current)

            x, y = current
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                if is_free_belief(nx, ny):
                    neighbor = (nx, ny)
                    tentative_g = g_score[current] + edge_cost(current, neighbor)
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def move_agent(self, new_pos):
        self.agent_pos = new_pos
        self.visited.add(new_pos)
        self.recent_positions.append(new_pos)
        self._increment_visit(new_pos)
        if len(self.recent_positions) >= 2 and self.recent_positions[-2] == new_pos:
            self.failure_count += 1

    def detect_cycle(self):
        if len(self.recent_positions) < 5:
            return False
        current = self.agent_pos
        count = sum(1 for p in self.recent_positions if p == current)
        return count >= 2

    def explore(self):
        x, y = self.agent_pos
        neighbors = self.world.get_free_neighbors(x, y)
        if not neighbors:
            return False

        in_cycle = self.detect_cycle()
        if in_cycle:
            unvisited = [n for n in neighbors if n not in self.visited]
            candidates = unvisited if unvisited else neighbors
        else:
            unvisited = [n for n in neighbors if n not in self.visited]
            candidates = unvisited if unvisited else neighbors

        best_score = -1e9
        best = None
        for pos in candidates:
            u = self.get_uncertainty(pos[0], pos[1])
            visits = self.get_visit_count(pos)
            novelty = self.novelty_weight * (1.0 / (1.0 + visits))
            penalty = self.revisit_penalty * visits
            score = u + novelty - penalty
            score += random.uniform(-0.05, 0.05)
            if in_cycle:
                score += random.uniform(-0.2, 0.2)
            if score > best_score:
                best_score = score
                best = pos

        if best is None:
            return False

        self.move_agent(best)
        return True

# ------------------------------------------------------------------------------
# 3. GUI WITH GIF RECORDING
# ------------------------------------------------------------------------------
class MazeSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Observation Maze Solver – Hierarchical Bayesian")
        self.maze = None
        self.pathfinder = None
        self.agent_pos = (0,0)
        self.goal = (0,0)

        self.viewport_cells = VIEWPORT_CELLS
        self.viewport_size_px = VIEWPORT_SIZE_PX
        self.cell_size = self.viewport_size_px // self.viewport_cells
        if self.cell_size < 2:
            self.cell_size = 2
        self.canvas_width = self.cell_size * self.viewport_cells
        self.canvas_height = self.canvas_width

        self.steps = 0
        self.current_path = []
        self.recording = False
        self.frames = []

        self.create_widgets()
        self.reset_maze()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        canvas_frame = ttk.LabelFrame(main_frame, text="Viewport (41×41, fixed)", padding=5)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.canvas = tk.Canvas(canvas_frame, bg='#1e2a3a',
                                width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=5)
        control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Movement
        move_frame = ttk.LabelFrame(control_frame, text="Move Agent", padding=3)
        move_frame.grid(row=0, column=0, columnspan=2, pady=3, sticky="ew")
        ttk.Button(move_frame, text="↑", command=lambda: self.move(0,-1)).grid(row=0, column=1)
        ttk.Button(move_frame, text="←", command=lambda: self.move(-1,0)).grid(row=1, column=0)
        ttk.Button(move_frame, text="→", command=lambda: self.move(1,0)).grid(row=1, column=2)
        ttk.Button(move_frame, text="↓", command=lambda: self.move(0,1)).grid(row=2, column=1)

        # Actions
        action_frame = ttk.LabelFrame(control_frame, text="Actions", padding=3)
        action_frame.grid(row=1, column=0, columnspan=2, pady=3, sticky="ew")
        ttk.Button(action_frame, text="Observe & Update", command=self.observe_step).pack(fill=tk.X, pady=1)
        ttk.Button(action_frame, text="Plan Path (A*)", command=self.plan_path).pack(fill=tk.X, pady=1)
        ttk.Button(action_frame, text="Step Solve", command=self.step_solve).pack(fill=tk.X, pady=1)
        ttk.Button(action_frame, text="Auto Solve", command=self.auto_solve).pack(fill=tk.X, pady=1)
        ttk.Button(action_frame, text="Solve & Record GIF", command=self.auto_solve_record).pack(fill=tk.X, pady=1)
        ttk.Button(action_frame, text="Reset", command=self.reset_maze).pack(fill=tk.X, pady=1)

        # Novelty sliders
        param_frame = ttk.LabelFrame(control_frame, text="Novelty Weights", padding=3)
        param_frame.grid(row=2, column=0, columnspan=2, pady=3, sticky="ew")
        ttk.Label(param_frame, text="Bonus:").grid(row=0, column=0, sticky="w")
        self.novelty_slider = tk.Scale(param_frame, from_=0.0, to=5.0, resolution=0.1,
                                       orient=tk.HORIZONTAL, length=80)
        self.novelty_slider.set(1.0)
        self.novelty_slider.grid(row=0, column=1, padx=2)
        ttk.Label(param_frame, text="Penalty:").grid(row=1, column=0, sticky="w")
        self.penalty_slider = tk.Scale(param_frame, from_=0.0, to=5.0, resolution=0.1,
                                       orient=tk.HORIZONTAL, length=80)
        self.penalty_slider.set(2.0)
        self.penalty_slider.grid(row=1, column=1, padx=2)

        # Status
        info_frame = ttk.LabelFrame(control_frame, text="Status", padding=3)
        info_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=5)
        self.info_text = tk.Text(info_frame, height=10, width=30,
                                 bg='#0a0f1a', fg='#b0c6ff', font=('Courier', 8))
        self.info_text.pack(fill=tk.BOTH, expand=True)

        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    # --------------------------------------------------------------------------
    # Viewport helpers
    # --------------------------------------------------------------------------
    def get_viewport_bounds(self):
        ax, ay = self.agent_pos
        half = self.viewport_cells // 2
        return ax - half, ay - half, ax + half, ay + half

    # --------------------------------------------------------------------------
    # Maze reset
    # --------------------------------------------------------------------------
    def reset_maze(self):
        self.maze = MazeGenerator(MAZE_SIZE, MAZE_SIZE)
        self.maze.generate()
        self.agent_pos = self.maze.start
        self.goal = self.maze.goal
        self.pathfinder = HierarchicalPathfinder(
            self.maze, self.agent_pos, num_layers=3,
            novelty_weight=self.novelty_slider.get(),
            revisit_penalty=self.penalty_slider.get()
        )
        self.steps = 0
        self.current_path = []
        self.frames = []
        self.update_display()
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "Maze ready. Observe, plan, explore.\n")

    # --------------------------------------------------------------------------
    # Movement & actions
    # --------------------------------------------------------------------------
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
        self.pathfinder.novelty_weight = self.novelty_slider.get()
        self.pathfinder.revisit_penalty = self.penalty_slider.get()
        path = self.pathfinder.plan_path(start=self.agent_pos, goal=self.goal, threshold=0.45)
        if path is None:
            self.info_text.insert(tk.END, "No path found (too uncertain).\n")
            self.current_path = []
        else:
            self.current_path = path
            self.info_text.insert(tk.END, f"A* path length: {len(path)} steps.\n")
        self.update_display()
        self.info_text.see(tk.END)

    def step_solve(self):
        self.pathfinder.novelty_weight = self.novelty_slider.get()
        self.pathfinder.revisit_penalty = self.penalty_slider.get()

        path = self.pathfinder.plan_path(start=self.agent_pos, goal=self.goal, threshold=0.45)
        moved = False
        if path is not None and len(path) >= 2:
            next_pos = path[1]
            if self.pathfinder.is_cell_free_belief(next_pos[0], next_pos[1], threshold=0.45):
                self.agent_pos = next_pos
                self.pathfinder.move_agent(next_pos)
                self.pathfinder.step(iterations=1)
                self.steps += 1
                moved = True
                self.current_path = path
            else:
                self.info_text.insert(tk.END, "Path blocked, exploring.\n")
        if not moved:
            if self.pathfinder.explore():
                self.agent_pos = self.pathfinder.agent_pos
                self.pathfinder.step(iterations=1)
                self.steps += 1
                moved = True
                self.current_path = []
                self.info_text.insert(tk.END, f"Explored to {self.agent_pos}\n")
            else:
                self.info_text.insert(tk.END, "Stuck: no free neighbor.\n")

        self.update_display()
        if self.agent_pos == self.goal:
            self.info_text.insert(tk.END, f"🎉 Goal reached in {self.steps} steps!\n")
        else:
            self.info_text.insert(tk.END, f"Step {self.steps}: at {self.agent_pos} (failures: {self.pathfinder.failure_count})\n")
        self.info_text.see(tk.END)

    # --------------------------------------------------------------------------
    # Auto solve with GIF recording
    # --------------------------------------------------------------------------
    def auto_solve(self, record=False, filename="solve.gif"):
        if record and not GIF_AVAILABLE:
            messagebox.showwarning("GIF", "imageio/PIL not installed. Cannot record.")
            record = False
        if record:
            self.frames = []
            self.recording = True
        self.info_text.insert(tk.END, "Auto-solving...\n")
        self.info_text.see(tk.END)
        self.pathfinder.novelty_weight = self.novelty_slider.get()
        self.pathfinder.revisit_penalty = self.penalty_slider.get()

        while self.agent_pos != self.goal:
            path = self.pathfinder.plan_path(start=self.agent_pos, goal=self.goal, threshold=0.45)
            moved = False
            if path is not None and len(path) >= 2:
                next_pos = path[1]
                if self.pathfinder.is_cell_free_belief(next_pos[0], next_pos[1], threshold=0.45):
                    self.agent_pos = next_pos
                    self.pathfinder.move_agent(next_pos)
                    self.pathfinder.step(iterations=1)
                    self.steps += 1
                    moved = True
                    self.current_path = path
                else:
                    self.info_text.insert(tk.END, "Path blocked, exploring.\n")
            if not moved:
                if self.pathfinder.explore():
                    self.agent_pos = self.pathfinder.agent_pos
                    self.pathfinder.step(iterations=1)
                    self.steps += 1
                    moved = True
                    self.current_path = []
                    self.info_text.insert(tk.END, f"Explored to {self.agent_pos}\n")
                else:
                    self.info_text.insert(tk.END, "Stuck! Cannot move.\n")
                    break

            self.update_display()
            if record:
                self.frames.append(self.render_to_pil())
            self.root.update()
            time.sleep(0.05)

        if self.agent_pos == self.goal:
            self.info_text.insert(tk.END, f"✅ Goal reached in {self.steps} steps!\n")
        else:
            self.info_text.insert(tk.END, "Failed.\n")
        self.info_text.see(tk.END)

        if record and self.frames:
            try:
                imageio.mimsave(filename, self.frames, fps=10)
                self.info_text.insert(tk.END, f"GIF saved as {filename}\n")
            except Exception as e:
                self.info_text.insert(tk.END, f"Error saving GIF: {e}\n")
            self.recording = False

    def auto_solve_record(self):
        self.auto_solve(record=True, filename="solve.gif")

    # --------------------------------------------------------------------------
    # Render to PIL Image (for GIF)
    # --------------------------------------------------------------------------
    def render_to_pil(self):
        min_x, min_y, max_x, max_y = self.get_viewport_bounds()
        vw = max_x - min_x + 1   # always VIEWPORT_CELLS
        vh = max_y - min_y + 1
        cell = self.cell_size
        img = Image.new('RGB', (vw*cell, vh*cell), '#000000')
        draw = ImageDraw.Draw(img)

        fused = self.pathfinder.get_fused_map()
        for y in range(min_y, max_y+1):
            for x in range(min_x, max_x+1):
                if 0 <= x < self.maze.width and 0 <= y < self.maze.height:
                    val = fused[y, x]
                    if val < 0.4:
                        color = (204,51,51)
                    elif val > 0.6:
                        color = (51,204,51)
                    else:
                        color = (102,102,136)
                else:
                    color = (0,0,0)
                px = (x - min_x) * cell
                py = (y - min_y) * cell
                draw.rectangle([px, py, px+cell-1, py+cell-1], fill=color)

        # Draw path
        for px, py in self.current_path:
            if min_x <= px <= max_x and min_y <= py <= max_y:
                cx = (px - min_x)*cell + cell//2
                cy = (py - min_y)*cell + cell//2
                draw.ellipse([cx-1, cy-1, cx+1, cy+1], fill=(255,255,255))

        # Goal
        gx, gy = self.goal
        if min_x <= gx <= max_x and min_y <= gy <= max_y:
            cx = (gx - min_x)*cell + cell//2
            cy = (gy - min_y)*cell + cell//2
            r = cell//3
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(0,255,255))

        # Agent
        ax, ay = self.agent_pos
        cx = (ax - min_x)*cell + cell//2
        cy = (ay - min_y)*cell + cell//2
        r = cell//3
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255,255,0))

        return img

    # --------------------------------------------------------------------------
    # Update display (Tkinter canvas)
    # --------------------------------------------------------------------------
    def update_display(self):
        self.canvas.delete("all")
        min_x, min_y, max_x, max_y = self.get_viewport_bounds()
        cell = self.cell_size

        fused = self.pathfinder.get_fused_map()
        for y in range(min_y, max_y+1):
            for x in range(min_x, max_x+1):
                if 0 <= x < self.maze.width and 0 <= y < self.maze.height:
                    val = fused[y, x]
                    if val < 0.4:
                        color = '#cc3333'
                    elif val > 0.6:
                        color = '#33cc33'
                    else:
                        color = '#666688'
                else:
                    color = '#000000'
                px = (x - min_x) * cell
                py = (y - min_y) * cell
                self.canvas.create_rectangle(px, py, px+cell, py+cell, fill=color, outline='')

        for px, py in self.current_path:
            if min_x <= px <= max_x and min_y <= py <= max_y:
                cx = (px - min_x)*cell + cell//2
                cy = (py - min_y)*cell + cell//2
                self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill='white', outline='white')

        gx, gy = self.goal
        if min_x <= gx <= max_x and min_y <= gy <= max_y:
            cx = (gx - min_x)*cell + cell//2
            cy = (gy - min_y)*cell + cell//2
            r = cell//3
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill='cyan', outline='white', width=2)

        ax, ay = self.agent_pos
        cx = (ax - min_x)*cell + cell//2
        cy = (ay - min_y)*cell + cell//2
        r = cell//3
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill='yellow', outline='white', width=2)

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"Agent: {self.agent_pos}\nGoal: {self.goal}\nSteps: {self.steps}\n")
        self.info_text.insert(tk.END, f"Failures: {self.pathfinder.failure_count}\n")
        if self.recording:
            self.info_text.insert(tk.END, "Recording GIF...\n")
        self.info_text.see(tk.END)

# ------------------------------------------------------------------------------
# 4. MAIN
# ------------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = MazeSolverGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
