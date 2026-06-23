"""
Deep Observation Maze Solver — Flask Public Server
3D Cube Maze · Hierarchical Bayesian Pathfinder
Streams rendered frames to all viewers in real time.
Run: python3 maze_flask.py
Open: http://localhost:8750
"""

import io, math, time, random, heapq, threading, base64, json
from collections import deque

import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify

try:
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
MAZE_N      = 11          # 3-D grid side (must be odd)
FRAME_W     = 700
FRAME_H     = 560
PORT        = 8750

# Isometric projection constants
ISO_TILE_W  = 36          # width of one iso cell (pixels)
ISO_TILE_H  = 18          # height of one iso cell
ISO_WALL_H  = 22          # height of wall cube face

# ──────────────────────────────────────────────────────────────
# 3-D MAZE GENERATOR  (recursive backtracker)
# ──────────────────────────────────────────────────────────────
class Maze3D:
    DIRS = [(2,0,0),(-2,0,0),(0,2,0),(0,-2,0),(0,0,2),(0,0,-2)]

    def __init__(self, n=MAZE_N):
        self.n = n
        self.grid = np.zeros((n,n,n), dtype=np.int8)
        self.start = (1,1,1)
        self.goal  = (n-2, n-2, n-2)

    def generate(self):
        n, g = self.n, self.grid
        g[:] = 0
        stack = [self.start]
        g[self.start] = 1
        while stack:
            x,y,z = stack[-1]
            nbrs = []
            for dx,dy,dz in self.DIRS:
                nx,ny,nz = x+dx,y+dy,z+dz
                if 0<=nx<n and 0<=ny<n and 0<=nz<n and g[nx,ny,nz]==0:
                    nbrs.append((nx,ny,nz,dx,dy,dz))
            if nbrs:
                nx,ny,nz,dx,dy,dz = random.choice(nbrs)
                g[x+dx//2, y+dy//2, z+dz//2] = 1
                g[nx,ny,nz] = 1
                stack.append((nx,ny,nz))
            else:
                stack.pop()
        g[self.start] = g[self.goal] = 1

    def is_free(self, x,y,z):
        n = self.n
        return 0<=x<n and 0<=y<n and 0<=z<n and self.grid[x,y,z]==1

    def free_neighbors(self, x,y,z):
        return [(x+dx,y+dy,z+dz)
                for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
                if self.is_free(x+dx,y+dy,z+dz)]

# ──────────────────────────────────────────────────────────────
# HIERARCHICAL BAYESIAN PATHFINDER  (original algorithm, 3-D)
# ──────────────────────────────────────────────────────────────
class HierarchicalPathfinder3D:
    def __init__(self, maze, novelty_weight=1.0, revisit_penalty=2.0):
        self.maze            = maze
        self.n               = maze.n
        self.agent           = list(maze.start)
        self.radii           = [4, 3, 2]
        self.noise_vars      = [0.20, 0.10, 0.05]
        sh = (self.n,)*3
        self.belief_maps     = [np.full(sh, 0.5) for _ in range(3)]
        self.visit_counts    = np.zeros(sh, dtype=int)
        self.visited         = set()
        self.recent          = deque(maxlen=20)
        self.failure_count   = 0
        self.novelty_weight  = novelty_weight
        self.revisit_penalty = revisit_penalty
        self.obs_count       = 0
        self.info_history    = []
        self.step_index      = 0
        self._inc(tuple(self.agent))

    def _inc(self, pos):
        x,y,z = pos
        if 0<=x<self.n and 0<=y<self.n and 0<=z<self.n:
            self.visit_counts[x,y,z] += 1

    # ── Observe ──────────────────────────────────────────────
    def observe(self):
        ax,ay,az = self.agent
        obs_list = []
        for li in range(3):
            r, nv = self.radii[li], self.noise_vars[li]
            om = np.full((self.n,)*3, 0.5)
            for dz in range(-r,r+1):
             for dy in range(-r,r+1):
              for dx in range(-r,r+1):
                nx,ny,nz = ax+dx,ay+dy,az+dz
                if 0<=nx<self.n and 0<=ny<self.n and 0<=nz<self.n:
                    tp = 1.0 if self.maze.is_free(nx,ny,nz) else 0.0
                    om[nx,ny,nz] = float(np.clip(tp + np.random.normal(0,math.sqrt(nv)),0,1))
            obs_list.append(om)
            self.obs_count += (2*r+1)**3
        return obs_list

    # ── Upward pass ──────────────────────────────────────────
    def upward_pass(self, observations):
        for i in range(3):
            pv = self.noise_vars[max(i-1,0)] * 0.5
            ov = self.noise_vars[i]
            prior = self.belief_maps[i-1] if i>0 else self.belief_maps[0]
            self.belief_maps[i] = (pv*observations[i] + ov*prior) / (pv+ov)

    # ── Downward pass ─────────────────────────────────────────
    def downward_pass(self):
        for i in range(1,-1,-1):
            self.belief_maps[i] = 0.3*self.belief_maps[i+1] + 0.7*self.belief_maps[i]

    def belief_step(self):
        obs = self.observe()
        self.upward_pass(obs)
        self.downward_pass()

    def fused(self):      return self.belief_maps[0]
    def uncertainty(self,x,y,z):
        v = float(self.belief_maps[0][x,y,z]); return v*(1-v)
    def is_free_belief(self,x,y,z,th=0.45):
        if not(0<=x<self.n and 0<=y<self.n and 0<=z<self.n): return False
        return float(self.belief_maps[0][x,y,z]) >= th

    # ── A* planner ────────────────────────────────────────────
    def plan_path(self, th=0.45):
        start = tuple(self.agent); goal = self.maze.goal
        h = lambda a,b: sum(abs(a[i]-b[i]) for i in range(3))
        def cost(nb):
            v = int(self.visit_counts[nb])
            return max(0.1, 1.0 + self.revisit_penalty*v - self.novelty_weight/(1+v))
        open_set=[]; heapq.heappush(open_set,(0.0,start))
        came_from={}; g={start:0.0}; closed=set()
        iters=0
        while open_set and iters<30000:
            _,cur = heapq.heappop(open_set); iters+=1
            if cur==goal:
                path=[]
                while cur in came_from: path.append(cur); cur=came_from[cur]
                path.append(start); path.reverse(); return path
            if cur in closed: continue
            closed.add(cur)
            x,y,z=cur
            for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nb=(x+dx,y+dy,z+dz)
                if self.is_free_belief(*nb,th):
                    tg=g[cur]+cost(nb)
                    if nb not in g or tg<g[nb]:
                        came_from[nb]=cur; g[nb]=tg
                        heapq.heappush(open_set,(tg+h(nb,goal),nb))
        return None

    def move_agent(self, pos):
        self.agent=list(pos); t=tuple(pos)
        self.visited.add(t); self.recent.append(t); self._inc(t)
        if len(self.recent)>=2 and self.recent[-2]==t: self.failure_count+=1

    def detect_cycle(self):
        if len(self.recent)<5: return False
        return sum(1 for p in self.recent if p==tuple(self.agent))>=2

    def explore(self):
        x,y,z=self.agent
        nbrs=self.maze.free_neighbors(x,y,z)
        if not nbrs: return False
        unv=[n for n in nbrs if tuple(n) not in self.visited]
        cands=unv if unv else nbrs
        in_cycle=self.detect_cycle()
        best_s,best=-1e9,None
        for pos in cands:
            v=int(self.visit_counts[pos])
            s=self.uncertainty(*pos)+self.novelty_weight/(1+v)-self.revisit_penalty*v
            s+=random.uniform(-0.2 if in_cycle else -0.05, 0.2 if in_cycle else 0.05)
            if s>best_s: best_s=s; best=pos
        if best is None: return False
        self.move_agent(best); return True

    # ── 7 Deep Observation metrics ────────────────────────────
    def deep_obs(self):
        fm=self.fused(); ax,ay,az=self.agent; n=self.n
        self.step_index+=1
        I_now=float(fm.mean()); self.info_history.append(I_now)

        # 1. Information Curvature  κ = d²I/dr²
        nbv=[]
        for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx,ny,nz=ax+dx,ay+dy,az+dz
            if 0<=nx<n and 0<=ny<n and 0<=nz<n: nbv.append(float(fm[nx,ny,nz]))
        cv=float(fm[ax,ay,az])
        kappa=sum(v-cv for v in nbv)/max(len(nbv),1)

        # 2. Observation Density  ρ = N_O/V
        rho_O=self.obs_count/n**3

        # 3. Observation Entropy  H = -Σ p log p
        flat=np.clip(fm.flatten(),1e-9,1-1e-9)
        H_O=float(-np.mean(flat*np.log2(flat)+(1-flat)*np.log2(1-flat)))

        # 4. Observation Momentum  M = dI/dt
        M_O=self.info_history[-1]-self.info_history[-2] if len(self.info_history)>=2 else 0.0

        # 5. Observation Acceleration  A = d²I/dt²
        A_O=(self.info_history[-1]-2*self.info_history[-2]+self.info_history[-3]
             if len(self.info_history)>=3 else 0.0)

        # 6. Recursive Self-Validation  Q = C·V·G
        C_n=1.0-self.uncertainty(ax,ay,az)
        vis_a=int(self.visit_counts[ax,ay,az])
        V_n=1.0/(1+vis_a)
        G_n=float(np.mean(fm>0.6))
        Q_n=C_n*V_n*G_n

        # 7. Meta-Observation  O_M = F(M)
        I_M=float(np.sum(np.abs(fm-0.5)))
        N_M=float(np.sum(1-np.abs(fm-0.5)*2))
        C_M=I_M/(I_M+N_M+1e-9)
        O_M=0.4*abs(M_O)+0.3*(1-H_O)+0.3*C_M

        # Verification between observers
        D_A=float(self.belief_maps[0][ax,ay,az])
        D_B=float(self.belief_maps[2][ax,ay,az])
        rho_corr=float(np.corrcoef(self.belief_maps[0].flatten(),
                                   self.belief_maps[2].flatten())[0,1])
        gamma_AB=1-rho_corr
        V_AB=gamma_AB*(1-abs(D_A-D_B))
        Gc=float(np.mean(fm>0.45))
        Pc=0.4*Gc+0.35*V_n+0.25*C_n

        return dict(
            step=self.step_index, I_now=round(I_now,5),
            agent_belief=round(cv,4), agent_unc=round(self.uncertainty(ax,ay,az),4),
            agent_visits=vis_a, global_consensus=round(Gc,4), overall_certainty=round(Pc,4),
            kappa=round(kappa,6), rho_O=round(rho_O,4),
            H_O=round(H_O,5), M_O=round(M_O,6), A_O=round(A_O,6),
            Q_n=round(Q_n,6), Q_accepted=bool(Q_n>0.01),
            C_n=round(C_n,4), V_n=round(V_n,4), G_n=round(G_n,4),
            I_M=round(I_M,4), N_M=round(N_M,4), C_M=round(C_M,5), O_M=round(O_M,5),
            d_AB=round(abs(D_A-D_B),5), gamma_AB=round(gamma_AB,4), V_AB=round(V_AB,5),
            S_A=round(D_A*self.step_index,4), S_B=round(D_B*self.step_index,4),
            r_m=round(2.0*math.log(10),4),
            I_history=[round(v,5) for v in self.info_history[-80:]],
            visited_count=len(self.visited),
            failures=self.failure_count,
        )

# ──────────────────────────────────────────────────────────────
# ISO RENDERER  (PIL → JPEG frames streamed as MJPEG)
# ──────────────────────────────────────────────────────────────

def _iso_project_t(gx, gy, gz, ox, oy, tw, th, wh):
    sx = ox + (gx - gy) * (tw // 2)
    sy = oy + (gx + gy) * (th // 2) - gz * wh
    return sx, sy


def _iso_bounds_t(n, ox, oy, tw, th, wh):
    """Return (min_x, min_y, max_x, max_y) of all cube corners."""
    xs, ys = [], []
    for gz in range(n):
        for gy in range(n):
            for gx in range(n):
                sx, sy = _iso_project_t(gx, gy, gz, ox, oy, tw, th, wh)
                xs.extend([sx - tw//2, sx + tw//2])
                ys.extend([sy, sy + th + wh])
    return min(xs), min(ys), max(xs), max(ys)


def _auto_origin_t(n, frame_w, frame_h, tw, th, wh, margin=20):
    """Compute ox, oy so the maze iso projection is centred in the frame."""
    trial_ox, trial_oy = frame_w // 2, frame_h // 2
    x0, y0, x1, y1 = _iso_bounds_t(n, trial_ox, trial_oy, tw, th, wh)
    cube_w = x1 - x0
    cube_h = y1 - y0
    ox = trial_ox + (frame_w - cube_w) // 2 - x0
    oy = trial_oy + (frame_h - cube_h) // 2 - y0
    return ox, oy


def _draw_cube_t(draw, gx, gy, gz, ox, oy, tw, th, wh,
                 fill_top, fill_left, fill_right, outline='#00ff88'):
    sx, sy = _iso_project_t(gx, gy, gz, ox, oy, tw, th, wh)
    top = [
        (sx,          sy),
        (sx + tw//2,  sy + th//2),
        (sx,          sy + th),
        (sx - tw//2,  sy + th//2),
    ]
    left = [top[2], top[3], (top[3][0], top[3][1]+wh), (top[2][0], top[2][1]+wh)]
    right = [top[1], top[2], (top[2][0], top[2][1]+wh), (top[1][0], top[1][1]+wh)]
    draw.polygon(left,  fill=fill_left,  outline=outline)
    draw.polygon(right, fill=fill_right, outline=outline)
    draw.polygon(top,   fill=fill_top,   outline=outline)


def _draw_sphere_t(draw, gx, gy, gz, ox, oy, tw, th, wh, color, r=7):
    sx, sy = _iso_project_t(gx, gy, gz, ox, oy, tw, th, wh)
    cx, cy = sx, sy + th//2 - wh//2
    for dr in range(r+4, 0, -1):
        alpha = int(80 * (1 - dr/(r+4)))
        draw.ellipse([cx-dr, cy-dr//2, cx+dr, cy+dr//2], fill=color[:3], outline=None)
    draw.ellipse([cx-r, cy-r//2, cx+r, cy+r//2], fill=color[:3], outline='white')


def render_frame(maze, solver, current_path, show_layer=-1):
    img  = Image.new('RGB', (FRAME_W, FRAME_H), '#05100a')
    draw = ImageDraw.Draw(img, 'RGBA')

    n  = maze.n
    fm = solver.fused()

    zoom_mode = show_layer != -1
    if zoom_mode:
        # Zoom: bigger tiles, draw only the selected layer
        tw, th, wh = 64, 32, 40
        sphere_r = 13
    else:
        tw, th, wh = ISO_TILE_W, ISO_TILE_H, ISO_WALL_H
        sphere_r = 8

    ox, oy = _auto_origin_t(n, FRAME_W, FRAME_H, tw, th, wh, margin=28)

    path_set = {tuple(p) for p in current_path}

    for z in range(n):
        if show_layer != -1 and z != show_layer:
            continue
        for y in range(n):
            for x in range(n):
                belief = float(fm[x,y,z])
                is_wall = maze.grid[x,y,z] == 0

                if is_wall:
                    bright = 0.35 + 0.65 * (1 - abs(belief - 0.5)*2)
                    r_t = int(0   * bright * 0.4 + 5  * 0.6)
                    g_t = int(255 * bright * 0.4 + 16 * 0.6)
                    b_t = int(136 * bright * 0.4 + 10 * 0.6)
                    r_l = int(r_t * 0.55); g_l = int(g_t * 0.55); b_l = int(b_t * 0.55)
                    r_r = int(r_t * 0.75); g_r = int(g_t * 0.75); b_r = int(b_t * 0.75)
                    out = '#%02x%02x%02x' % (int(min(g_t*0.3+40,255)), int(min(g_t,255)), int(min(g_t*0.5,255)))
                    _draw_cube_t(draw, x, y, z, ox, oy, tw, th, wh,
                                 (r_t,g_t,b_t), (r_l,g_l,b_l), (r_r,g_r,b_r), outline=out)
                else:
                    _draw_cube_t(draw, x, y, z, ox, oy, tw, th, wh,
                                 '#0d1f14', '#081208', '#0a1a0f', outline='#0f2a14')
                    if (x,y,z) in path_set:
                        sx, sy = _iso_project_t(x, y, z, ox, oy, tw, th, wh)
                        pr = max(3, tw // 10)
                        cx, cy = sx, sy + th//2
                        draw.ellipse([cx-pr, cy-pr//2, cx+pr, cy+pr//2], fill='white')

    sx2,sy2,sz2 = maze.start
    if show_layer==-1 or show_layer==sz2:
        _draw_sphere_t(draw, sx2,sy2,sz2, ox,oy, tw,th,wh, (255,30,50), r=sphere_r)

    gx2,gy2,gz2 = maze.goal
    if show_layer==-1 or show_layer==gz2:
        _draw_sphere_t(draw, gx2,gy2,gz2, ox,oy, tw,th,wh, (190,50,255), r=sphere_r)

    ax,ay,az = solver.agent
    if show_layer==-1 or show_layer==az:
        _draw_sphere_t(draw, ax,ay,az, ox,oy, tw,th,wh, (255,225,30), r=max(sphere_r-1,6))

    _draw_hud(draw, solver, current_path, show_layer)
    return img

def _draw_hud(draw, solver, path, show_layer=-1):
    ax,ay,az = solver.agent
    lines = [
        f"Agent  ({ax},{ay},{az})",
        f"Goal   {solver.maze.goal}",
        f"Steps  {solver.step_index}",
        f"Path   {len(path)} cells",
        f"Visits {len(solver.visited)}",
        f"Fails  {solver.failure_count}",
    ]
    if show_layer != -1:
        lines.insert(0, f"ZOOM   Z={show_layer}")
    x0, y0 = 10, 10
    for i, line in enumerate(lines):
        draw.text((x0+1, y0+i*14+1), line, fill='#000000')
        draw.text((x0,   y0+i*14),   line, fill='#00ff88')

# ──────────────────────────────────────────────────────────────
# GLOBAL SOLVER STATE  (protected by a lock)
# ──────────────────────────────────────────────────────────────
lock         = threading.Lock()
maze         = None
solver       = None
current_path = []
steps        = 0
auto_running = False
auto_thread  = None
show_layer   = -1
recording    = False
gif_frames   = []
last_frame   = None      # cached PIL Image

def _new_maze(nov=1.0, pen=2.0):
    global maze, solver, current_path, steps, auto_running, recording, gif_frames
    auto_running = False
    recording    = False
    gif_frames   = []
    maze         = Maze3D(MAZE_N)
    maze.generate()
    solver       = HierarchicalPathfinder3D(maze, nov, pen)
    solver.belief_step()
    current_path = []
    steps        = 0

def _get_frame():
    global last_frame
    with lock:
        img = render_frame(maze, solver, current_path, show_layer)
    last_frame = img
    return img

def _do_step(nov=None, pen=None):
    global current_path, steps
    with lock:
        if nov is not None: solver.novelty_weight  = nov
        if pen is not None: solver.revisit_penalty = pen
        path = solver.plan_path()
        moved = False
        if path and len(path)>=2:
            nxt = path[1]
            if solver.is_free_belief(*nxt):
                solver.move_agent(nxt)
                solver.belief_step()
                current_path = path
                steps += 1
                moved = True
        if not moved:
            solver.explore()
            solver.belief_step()
            current_path = []
            steps += 1
        reached = tuple(solver.agent) == maze.goal
    return reached

def _auto_loop():
    global auto_running, gif_frames, recording
    while auto_running:
        done = _do_step()
        if recording:
            img = _get_frame()
            gif_frames.append(img.copy())
        if done:
            auto_running = False
            break
        time.sleep(0.08)

# ──────────────────────────────────────────────────────────────
# FLASK APP
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── MJPEG stream ──────────────────────────────────────────────
def mjpeg_generator():
    while True:
        if maze is None:
            time.sleep(0.1)
            continue
        img  = _get_frame()
        buf  = io.BytesIO()
        img.save(buf, format='JPEG', quality=82)
        frame_bytes = buf.getvalue()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + frame_bytes + b'\r\n')
        time.sleep(1/24)  # ~24 fps

@app.route('/stream')
def stream():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ── API endpoints ─────────────────────────────────────────────
@app.route('/api/reset', methods=['POST'])
def api_reset():
    d = request.get_json(silent=True) or {}
    _new_maze(float(d.get('novelty',1.0)), float(d.get('penalty',2.0)))
    _get_frame()
    return jsonify({'ok': True})

@app.route('/api/step', methods=['POST'])
def api_step():
    d = request.get_json(silent=True) or {}
    reached = _do_step(float(d.get('novelty',1.0)), float(d.get('penalty',2.0)))
    with lock:
        deep = solver.deep_obs()
    return jsonify({'reached': reached, 'deep': deep,
                    'path_len': len(current_path), 'steps': steps})

@app.route('/api/auto/start', methods=['POST'])
def api_auto_start():
    global auto_running, auto_thread
    d = request.get_json(silent=True) or {}
    if maze is None: _new_maze()
    with lock:
        solver.novelty_weight  = float(d.get('novelty',1.0))
        solver.revisit_penalty = float(d.get('penalty',2.0))
    auto_running = True
    auto_thread  = threading.Thread(target=_auto_loop, daemon=True)
    auto_thread.start()
    return jsonify({'ok': True})

@app.route('/api/auto/stop', methods=['POST'])
def api_auto_stop():
    global auto_running
    auto_running = False
    return jsonify({'ok': True})

@app.route('/api/observe', methods=['POST'])
def api_observe():
    with lock:
        solver.belief_step()
        deep = solver.deep_obs()
    return jsonify({'ok': True, 'deep': deep})

@app.route('/api/plan', methods=['POST'])
def api_plan():
    global current_path
    with lock:
        path = solver.plan_path()
        current_path = path if path else []
        deep = solver.deep_obs()
    return jsonify({'path_len': len(current_path), 'deep': deep})

@app.route('/api/layer', methods=['POST'])
def api_layer():
    global show_layer
    d = request.get_json(silent=True) or {}
    show_layer = int(d.get('layer', -1))
    return jsonify({'ok': True})

@app.route('/api/record/start', methods=['POST'])
def api_record_start():
    global recording, gif_frames
    recording  = True
    gif_frames = []
    return jsonify({'ok': True})

@app.route('/api/record/stop', methods=['POST'])
def api_record_stop():
    global recording
    recording = False
    if not PIL_OK or not gif_frames:
        return jsonify({'ok': False, 'error': 'No frames or PIL unavailable'})
    buf = io.BytesIO()
    gif_frames[0].save(buf, format='GIF', save_all=True,
                       append_images=gif_frames[1:], duration=80, loop=0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({'ok': True, 'gif_b64': b64, 'frames': len(gif_frames)})

@app.route('/api/deep', methods=['GET'])
def api_deep():
    if maze is None: return jsonify({})
    with lock:
        d = solver.deep_obs()
    return jsonify(d)

@app.route('/api/status', methods=['GET'])
def api_status():
    if maze is None: return jsonify({'ready': False})
    with lock:
        ag = list(solver.agent)
        goal = list(maze.goal)
        reached = tuple(solver.agent)==maze.goal
    return jsonify({'ready': True, 'agent': ag, 'goal': goal,
                    'steps': steps, 'auto': auto_running,
                    'recording': recording, 'reached': reached,
                    'path_len': len(current_path)})

# ── Main page ─────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

# ──────────────────────────────────────────────────────────────
# HTML / JS FRONTEND
# ──────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deep Observation — 3D Cube Maze Solver</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#050a08;--surf:#0b1510;--bdr:#0d2a18;
  --grn:#00ff88;--gdim:#00884a;--red:#ff2244;
  --vio:#cc44ff;--yel:#ffe055;--blu:#44aaff;
  --txt:#c8e8d8;--mut:#446655;
  --mono:'Space Mono',monospace;--body:'Inter',sans-serif;
}
html,body{height:100%;background:var(--bg);color:var(--txt);font-family:var(--body);overflow:hidden}

/* ── SHELL ── */
#app{display:grid;grid-template-rows:50px 1fr;grid-template-columns:1fr 330px;height:100vh}
header{grid-column:1/-1;display:flex;align-items:center;gap:14px;padding:0 18px;
  border-bottom:1px solid var(--bdr);background:var(--surf)}
header h1{font-family:var(--mono);font-size:12px;font-weight:700;letter-spacing:.12em;
  color:var(--grn);text-transform:uppercase}
.sub{font-size:10px;color:var(--mut);font-family:var(--mono)}
.badge{margin-left:auto;font-family:var(--mono);font-size:9px;color:var(--mut);
  border:1px solid var(--bdr);padding:3px 8px;border-radius:3px}

/* ── VIEWER ── */
#viewer{position:relative;overflow:hidden;background:#020704;display:flex;align-items:center;justify-content:center}
#stream-img{max-width:100%;max-height:100%;object-fit:contain;display:block}
#overlay{position:absolute;top:0;left:0;right:0;bottom:0;pointer-events:none}
#reached-banner{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-family:var(--mono);font-size:22px;color:var(--grn);text-shadow:0 0 20px var(--grn);
  display:none;text-align:center;line-height:1.5}
#layer-bar{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);
  display:flex;gap:4px;background:rgba(5,10,8,.92);border:1px solid var(--bdr);
  border-radius:5px;padding:5px 8px;flex-wrap:wrap;max-width:95%}
#layer-bar span{font-family:var(--mono);font-size:9px;color:var(--mut);align-self:center;margin-right:3px}
.lbtn{font-family:var(--mono);font-size:9px;padding:3px 7px;border:1px solid var(--bdr);
  background:transparent;color:var(--mut);border-radius:2px;cursor:pointer;transition:all .12s}
.lbtn.on,.lbtn:hover{border-color:var(--grn);color:var(--grn);background:rgba(0,255,136,.07)}
#rec-dot{position:absolute;top:10px;right:12px;width:9px;height:9px;border-radius:50%;
  background:var(--red);box-shadow:0 0 8px var(--red);display:none;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.2}}

/* ── PANEL ── */
#panel{border-left:1px solid var(--bdr);background:var(--surf);display:flex;flex-direction:column;overflow:hidden}
.psec{border-bottom:1px solid var(--bdr);padding:11px 13px;flex-shrink:0}
.plbl{font-family:var(--mono);font-size:8px;letter-spacing:.15em;color:var(--mut);
  text-transform:uppercase;margin-bottom:7px}
.brow{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:5px}
.btn{flex:1;font-family:var(--mono);font-size:9px;padding:6px 6px;border:1px solid var(--bdr);
  background:transparent;color:var(--txt);border-radius:3px;cursor:pointer;transition:all .12s;white-space:nowrap}
.btn:hover{border-color:var(--grn);color:var(--grn);background:rgba(0,255,136,.04)}
.btn.pri{border-color:var(--gdim);color:var(--grn)}
.btn.rec-on{border-color:var(--red)!important;color:var(--red)!important;background:rgba(255,34,68,.08)!important}
.srow{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.srow label{font-family:var(--mono);font-size:8px;color:var(--mut);width:68px;flex-shrink:0}
.srow input[type=range]{flex:1;accent-color:var(--grn)}
.val{font-family:var(--mono);font-size:9px;color:var(--grn);width:24px;text-align:right}
#status{font-family:var(--mono);font-size:9px;color:var(--mut);margin-top:5px;min-height:13px}

/* ── MATH PANEL ── */
#math-wrap{flex:1;overflow-y:auto;padding:9px 12px 14px}
.tog-bar{display:flex;gap:3px;flex-wrap:wrap;margin-bottom:9px}
.mtog{font-family:var(--mono);font-size:8px;padding:3px 6px;border:1px solid var(--bdr);
  background:transparent;color:var(--mut);border-radius:2px;cursor:pointer;transition:all .12s}
.mtog.on{border-color:var(--grn);color:var(--grn);background:rgba(0,255,136,.08)}
.mblk{margin-bottom:9px;border:1px solid var(--bdr);border-radius:4px;overflow:hidden}
.mblk.hide{display:none}
.mhd{font-family:var(--mono);font-size:8px;letter-spacing:.08em;color:var(--gdim);
  background:rgba(0,255,136,.04);padding:4px 9px;border-bottom:1px solid var(--bdr);
  text-transform:uppercase;display:flex;align-items:center;gap:6px}
.dot-n{width:7px;height:7px;border-radius:50%;background:var(--grn);box-shadow:0 0 5px var(--grn);flex-shrink:0}
.dot-c{width:7px;height:7px;border-radius:50%;background:var(--blu);box-shadow:0 0 4px var(--blu);flex-shrink:0}
.mbdy{padding:7px 10px;font-family:var(--mono);font-size:9px;line-height:1.75;color:#8bbba0}
.eq{display:block;color:#b8eecf;margin:2px 0;padding-left:5px;border-left:2px solid var(--gdim)}
.lv{color:var(--grn);font-weight:700}
.lv-r{color:var(--red);font-weight:700}
.lv-v{color:var(--vio);font-weight:700}
.lv-y{color:var(--yel);font-weight:700}
.lv-b{color:var(--blu);font-weight:700}
.ok{color:var(--grn);font-weight:700}
.fail{color:var(--red);font-weight:700}
.dim{color:var(--mut)}
.spark{width:100%;height:28px;margin-top:3px}

/* legend */
.leg{display:flex;flex-direction:column;gap:4px}
.lrow{display:flex;align-items:center;gap:7px;font-size:9px;font-family:var(--mono);color:var(--mut)}
.ldot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.lsw{width:9px;height:9px;border-radius:2px;flex-shrink:0}

/* gif download */
#gif-link{display:none;font-family:var(--mono);font-size:9px;color:var(--grn);
  text-decoration:underline;cursor:pointer;margin-top:4px}
</style>
</head>
<body>
<div id="app">

<header>
  <h1>Deep Observation — 3D Cube Maze</h1>
  <span class="sub">Hierarchical Bayesian · A* · 7-Module Math Suite</span>
  <span class="badge">:8750 · MJPEG Live</span>
</header>

<!-- ── VIEWER ── -->
<div id="viewer">
  <img id="stream-img" src="/stream" alt="maze stream">
  <div id="overlay">
    <div id="reached-banner">🎉 GOAL REACHED!<br><span style="font-size:13px" id="reached-steps"></span></div>
    <div id="rec-dot"></div>
  </div>
  <div id="layer-bar">
    <span>Z-LAYER</span>
    <button class="lbtn on" onclick="setLayer(-1)">ALL</button>
    <button class="lbtn" onclick="setLayer(1)">Z1</button>
    <button class="lbtn" onclick="setLayer(3)">Z3</button>
    <button class="lbtn" onclick="setLayer(5)">Z5</button>
    <button class="lbtn" onclick="setLayer(7)">Z7</button>
    <button class="lbtn" onclick="setLayer(9)">Z9</button>
  </div>
</div>

<!-- ── PANEL ── -->
<div id="panel">

  <!-- Controls -->
  <div class="psec">
    <div class="plbl">Controls</div>
    <div class="brow">
      <button class="btn pri" onclick="doReset()">⟳ New Maze</button>
      <button class="btn" onclick="doObserve()">👁 Observe</button>
      <button class="btn" onclick="doPlan()">📐 Plan</button>
    </div>
    <div class="brow">
      <button class="btn" onclick="doStep()">▷ Step</button>
      <button class="btn" id="abtn" onclick="toggleAuto()">▶ Auto Solve</button>
    </div>
    <div class="brow">
      <button class="btn" id="rbtn" onclick="toggleRecord()">⏺ Record GIF</button>
      <a id="gif-link" download="maze_solve.gif">⬇ Download GIF</a>
    </div>
    <div class="srow"><label>Novelty η</label>
      <input type="range" id="nov" min="0" max="5" step="0.1" value="1.0" oninput="sv(this,'nv')">
      <span class="val" id="nv">1.0</span></div>
    <div class="srow"><label>Penalty λ</label>
      <input type="range" id="pen" min="0" max="5" step="0.1" value="2.0" oninput="sv(this,'pv')">
      <span class="val" id="pv">2.0</span></div>
    <div class="srow"><label>Poll rate</label>
      <input type="range" id="poll" min="100" max="2000" step="50" value="300" oninput="sv(this,'pollv');restartPoll()">
      <span class="val" id="pollv">300</span><span class="dim" style="font-size:8px">ms</span></div>
    <div id="status">Loading…</div>
  </div>

  <!-- Legend -->
  <div class="psec">
    <div class="plbl">Legend</div>
    <div class="leg">
      <div class="lrow"><span class="ldot" style="background:#ff2244;box-shadow:0 0 6px #ff2244"></span>Start (1,1,1)</div>
      <div class="lrow"><span class="ldot" style="background:#cc44ff;box-shadow:0 0 6px #cc44ff"></span>Goal (N-2,N-2,N-2)</div>
      <div class="lrow"><span class="ldot" style="background:#ffe055;box-shadow:0 0 6px #ffe055"></span>Agent</div>
      <div class="lrow"><span class="ldot" style="background:#fff"></span>A* path</div>
      <div class="lrow"><span class="lsw" style="background:rgba(0,255,136,.4);border:1px solid #00ff88"></span>Wall cube (40% α glow)</div>
      <div class="lrow"><span class="lsw" style="background:#0d2015;border:1px solid #112211"></span>Open corridor</div>
    </div>
  </div>

  <!-- Math Toggle Bar -->
  <div class="psec" style="padding-bottom:6px">
    <div class="plbl">Math Suite — Toggle Modules</div>
    <div class="tog-bar" id="tog-bar"></div>
  </div>

  <!-- Math Blocks -->
  <div id="math-wrap">

    <div class="mblk" id="blk-observe">
      <div class="mhd"><span class="dot-c"></span>1 · Noise Interaction Field</div>
      <div class="mbdy">
        <span class="eq">obs(x) = P_true + 𝒩(0, σ²)</span>
        σ² = <span class="lv">0.20, 0.10, 0.05</span> (layers 0-2)<br>
        Obs accumulated: <span class="lv" id="d-obs">—</span>
      </div>
    </div>

    <div class="mblk" id="blk-upward">
      <div class="mhd"><span class="dot-c"></span>2 · Upward Pass</div>
      <div class="mbdy">
        <span class="eq">P(free|obs) = (σ²_p·obs + σ²_o·prior) / (σ²_p+σ²_o)</span>
        D_A (local): <span class="lv" id="d-DA">—</span> &nbsp;
        D_B (global): <span class="lv" id="d-DB">—</span><br>
        S_A=<span class="lv" id="d-SA">—</span> &nbsp; S_B=<span class="lv" id="d-SB">—</span>
      </div>
    </div>

    <div class="mblk" id="blk-down">
      <div class="mhd"><span class="dot-c"></span>3 · Downward Pass / Consensus</div>
      <div class="mbdy">
        <span class="eq">b[i] = 0.3·b[i+1] + 0.7·b[i]</span>
        Global Consensus G: <span class="lv" id="d-G">—</span><br>
        <span class="eq">P_c = 0.4G + 0.35V + 0.25C</span>
        Overall Certainty P_c: <span class="lv" id="d-Pc">—</span>
      </div>
    </div>

    <div class="mblk" id="blk-astar">
      <div class="mhd"><span class="dot-c"></span>4 · A* Planner</div>
      <div class="mbdy">
        <span class="eq">h(n) = |Δx|+|Δy|+|Δz|  (3D Manhattan)</span>
        <span class="eq">cost(n) = 1 + λ·v(n) − η/(1+v(n))</span>
        λ=<span class="lv" id="d-lam">—</span> η=<span class="lv" id="d-eta">—</span><br>
        Path: <span class="lv" id="d-plen">—</span> cells &nbsp; Failures: <span class="lv-r" id="d-fail">—</span>
      </div>
    </div>

    <div class="mblk" id="blk-verify">
      <div class="mhd"><span class="dot-c"></span>5 · Verification Between Observers</div>
      <div class="mbdy">
        <span class="eq">d_AB=‖D_A−D_B‖= <span id="d-dAB">—</span></span>
        <span class="eq">γ_AB=1−ρ_AB= <span id="d-gAB">—</span></span>
        <span class="eq">V_AB=γ_AB(1−|C_A−C_B|)= <span id="d-VAB">—</span></span>
        r_m: <span class="lv" id="d-rm">—</span>
        <span class="dim"> = (1/λ)ln(I₀/I_th)</span>
      </div>
    </div>

    <!-- NEW ADDITIONS -->
    <div class="mblk" id="blk-curvature">
      <div class="mhd"><span class="dot-n"></span>NEW · Information Curvature</div>
      <div class="mbdy">
        Bending of info through medium. Indicates barriers &amp; hidden structures.<br>
        <span class="eq">κᵢ = d²I/dr² ≈ Laplacian(belief) @ agent</span>
        κᵢ = <span class="lv" id="d-kappa">—</span><br>
        <span class="dim">+ve=converging &nbsp; −ve=diverging</span>
      </div>
    </div>

    <div class="mblk" id="blk-density">
      <div class="mhd"><span class="dot-n"></span>NEW · Observation Density</div>
      <div class="mbdy">
        Observations per unit volume. Indicates space coverage.<br>
        <span class="eq">ρ_O = N_O / V</span>
        N_O=<span class="lv" id="d-NO">—</span> V=<span class="lv" id="d-V">—</span><br>
        ρ_O=<span class="lv" id="d-rho">—</span>
      </div>
    </div>

    <div class="mblk" id="blk-entropy">
      <div class="mhd"><span class="dot-n"></span>NEW · Observation Entropy</div>
      <div class="mbdy">
        High=unknown · Low=understood<br>
        <span class="eq">H_O = −Σ pᵢ log₂ pᵢ (avg per cell)</span>
        H_O = <span class="lv" id="d-HO">—</span> bits
        <canvas class="spark" id="sp-H"></canvas>
      </div>
    </div>

    <div class="mblk" id="blk-momentum">
      <div class="mhd"><span class="dot-n"></span>NEW · Observation Momentum</div>
      <div class="mbdy">
        Rate of information accumulation. Indicates emerging events.<br>
        <span class="eq">M_O = dI/dt</span>
        I(t)=<span class="lv" id="d-It">—</span> &nbsp; M_O=<span class="lv" id="d-MO">—</span>
        <canvas class="spark" id="sp-I"></canvas>
      </div>
    </div>

    <div class="mblk" id="blk-accel">
      <div class="mhd"><span class="dot-n"></span>NEW · Observation Acceleration</div>
      <div class="mbdy">
        Rate of change of momentum. Detects surges &amp; anomalies.<br>
        <span class="eq">A_O = d²I/dt²</span>
        A_O=<span class="lv" id="d-AO">—</span><br>
        <span id="d-surge" class="dim">—</span>
      </div>
    </div>

    <div class="mblk" id="blk-validation">
      <div class="mhd"><span class="dot-n"></span>NEW · Recursive Self-Validation</div>
      <div class="mbdy">
        Every addition observed, verified &amp; tested before acceptance.<br>
        <span class="eq">Q_n = C_n · V_n · G_n</span>
        C_n=<span class="lv" id="d-Cn">—</span>
        V_n=<span class="lv" id="d-Vn">—</span>
        G_n=<span class="lv" id="d-Gn">—</span><br>
        Q_n=<span class="lv" id="d-Qn">—</span> &nbsp; τ_Q=<span class="dim">0.01</span><br>
        <span id="d-Qst">—</span>
      </div>
    </div>

    <div class="mblk" id="blk-meta">
      <div class="mhd"><span class="dot-n"></span>NEW · Meta-Observation Layer</div>
      <div class="mbdy">
        Observes the framework itself for continual improvement.<br>
        <span class="eq">O_M = F(M) = 0.4|M_O| + 0.3(1−H_O) + 0.3·C_M</span>
        <span class="eq">C_M = I_M / (I_M + N_M)</span>
        I_M=<span class="lv" id="d-IM">—</span>
        N_M=<span class="lv" id="d-NM">—</span>
        C_M=<span class="lv" id="d-CM">—</span><br>
        O_M=<span class="lv-v" id="d-OM">—</span>
      </div>
    </div>

    <div class="mblk" id="blk-live">
      <div class="mhd"><span class="dot-c"></span>Live State</div>
      <div class="mbdy">
        Agent: <span class="lv-y" id="d-ag">—</span><br>
        Goal:  <span class="lv-v" id="d-goal">—</span><br>
        Steps: <span class="lv" id="d-steps">0</span>
        &nbsp; Visited: <span class="lv" id="d-vis">—</span><br>
        Belief@agent: <span class="lv" id="d-bel">—</span><br>
        Uncertainty:  <span class="lv" id="d-unc">—</span>
      </div>
    </div>

  </div><!-- math-wrap -->
</div><!-- panel -->
</div><!-- app -->

<script>
// ── Module toggle ─────────────────────────────────────────
const MODS = [
  {id:'observe',   label:'Noise Field', on:true},
  {id:'upward',    label:'Upward',      on:true},
  {id:'down',      label:'Consensus',   on:true},
  {id:'astar',     label:'A*',          on:true},
  {id:'verify',    label:'Verify',      on:true},
  {id:'curvature', label:'κ Curve',     on:true},
  {id:'density',   label:'ρ Density',   on:true},
  {id:'entropy',   label:'H Entropy',   on:true},
  {id:'momentum',  label:'M Moment',    on:true},
  {id:'accel',     label:'A Accel',     on:true},
  {id:'validation',label:'Q Valid',     on:true},
  {id:'meta',      label:'Meta-Obs',    on:true},
  {id:'live',      label:'Live State',  on:true},
];
const togBar = document.getElementById('tog-bar');
MODS.forEach(m => {
  const b = document.createElement('button');
  b.className = 'mtog on'; b.textContent = m.label;
  b.onclick = () => {
    m.on = !m.on; b.classList.toggle('on', m.on);
    document.getElementById('blk-'+m.id).classList.toggle('hide', !m.on);
  };
  togBar.appendChild(b);
});

// ── Slider helper ─────────────────────────────────────────
function sv(el, id) { document.getElementById(id).textContent = parseFloat(el.value).toFixed(1); }
function nov() { return parseFloat(document.getElementById('nov').value); }
function pen() { return parseFloat(document.getElementById('pen').value); }
function setStatus(t) { document.getElementById('status').textContent = t; }

// ── API helpers ───────────────────────────────────────────
async function post(url, body={}) {
  const r = await fetch(url, {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  return r.json();
}

// ── Controls ──────────────────────────────────────────────
async function doReset() {
  setStatus('Generating 3D maze…');
  document.getElementById('reached-banner').style.display='none';
  await post('/api/reset', {novelty:nov(), penalty:pen()});
  setStatus('Ready.');
}
async function doObserve() {
  const d = await post('/api/observe');
  updateMath(d.deep); setStatus('Observed & updated beliefs.');
}
async function doPlan() {
  const d = await post('/api/plan');
  updateMath(d.deep); setStatus(`Planned: ${d.path_len} cells.`);
}
async function doStep() {
  const d = await post('/api/step', {novelty:nov(), penalty:pen()});
  updateMath(d.deep);
  if (d.reached) showReached(d.deep?.step);
  else setStatus(`Step ${d.steps} · path ${d.path_len} cells`);
}

let autoOn = false;
async function toggleAuto() {
  if (autoOn) {
    await post('/api/auto/stop');
    autoOn = false;
    document.getElementById('abtn').textContent = '▶ Auto Solve';
    setStatus('Stopped.');
  } else {
    await post('/api/auto/start', {novelty:nov(), penalty:pen()});
    autoOn = true;
    document.getElementById('abtn').textContent = '⏹ Stop';
    setStatus('Auto solving…');
  }
}

let recOn = false;
async function toggleRecord() {
  const btn = document.getElementById('rbtn');
  const dot = document.getElementById('rec-dot');
  if (!recOn) {
    await post('/api/record/start');
    recOn = true;
    btn.classList.add('rec-on');
    btn.textContent = '⏹ Stop Recording';
    dot.style.display = 'block';
    document.getElementById('gif-link').style.display = 'none';
    setStatus('Recording GIF…');
  } else {
    setStatus('Encoding GIF…');
    const d = await post('/api/record/stop');
    recOn = false;
    btn.classList.remove('rec-on');
    btn.textContent = '⏺ Record GIF';
    dot.style.display = 'none';
    if (d.ok && d.gif_b64) {
      const link = document.getElementById('gif-link');
      link.href = 'data:image/gif;base64,' + d.gif_b64;
      link.style.display = 'block';
      setStatus(`GIF ready — ${d.frames} frames. Click to download.`);
    } else {
      setStatus('GIF encode failed (need Pillow+imageio).');
    }
  }
}

function setLayer(z) {
  post('/api/layer', {layer:z});
  document.querySelectorAll('.lbtn').forEach(b => {
    const t = b.textContent;
    b.classList.toggle('on', (z===-1&&t==='ALL')||(z!==-1&&t==='Z'+z));
  });
}

function showReached(steps) {
  const b = document.getElementById('reached-banner');
  document.getElementById('reached-steps').textContent = `${steps} steps`;
  b.style.display = 'block';
  autoOn = false;
  document.getElementById('abtn').textContent = '▶ Auto Solve';
  setStatus(`🎉 Goal reached in ${steps} steps!`);
}

// ── Status polling ────────────────────────────────────────
let pollTimer = null;
function restartPoll() {
  clearInterval(pollTimer);
  const ms = parseInt(document.getElementById('poll').value);
  pollTimer = setInterval(pollStatus, ms);
}
async function pollStatus() {
  try {
    const s = await fetch('/api/status').then(r=>r.json());
    if (!s.ready) return;
    if (s.reached && !autoOn) showReached(s.steps);
    if (!s.auto && autoOn) {
      autoOn = false;
      document.getElementById('abtn').textContent = '▶ Auto Solve';
    }
    // also pull deep math
    const d = await fetch('/api/deep').then(r=>r.json());
    updateMath(d);
  } catch(e) {}
}
restartPoll();

// ── Sparkline history ─────────────────────────────────────
const hI=[], hH=[];
function spark(id, data, color) {
  const cv = document.getElementById(id);
  if (!cv || data.length < 2) return;
  const ctx = cv.getContext('2d');
  cv.width = cv.offsetWidth * devicePixelRatio;
  cv.height = cv.offsetHeight * devicePixelRatio;
  ctx.clearRect(0,0,cv.width,cv.height);
  const mn=Math.min(...data), mx=Math.max(...data), rng=mx-mn||0.001;
  const W=cv.width, H=cv.height;
  ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.beginPath();
  data.forEach((v,i)=>{
    const x=(i/(data.length-1))*W, y=H-((v-mn)/rng)*(H*.8)-H*.1;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();
}

// ── Math panel update ─────────────────────────────────────
function set(id,v){ const e=document.getElementById(id); if(e) e.textContent=v??'—'; }
function updateMath(d) {
  if (!d || !Object.keys(d).length) return;

  // Core
  set('d-obs',  Math.round(d.rho_O * 1331));
  set('d-DA',   d.S_A !== undefined ? (d.I_now||0).toFixed(4) : '—');
  set('d-DB',   d.d_AB !== undefined ? ((d.I_now||0)+d.d_AB).toFixed(4) : '—');
  set('d-SA',   d.S_A);  set('d-SB', d.S_B);
  set('d-G',    d.global_consensus); set('d-Pc', d.overall_certainty);
  set('d-lam',  parseFloat(document.getElementById('pen').value).toFixed(1));
  set('d-eta',  parseFloat(document.getElementById('nov').value).toFixed(1));
  set('d-plen', d.visited_count ? '—' : '—');
  set('d-fail', d.failures);
  set('d-dAB',  d.d_AB); set('d-gAB', d.gamma_AB); set('d-VAB', d.V_AB);
  set('d-rm',   d.r_m);

  // New additions
  set('d-kappa', d.kappa);
  set('d-NO',    Math.round((d.rho_O||0)*1331));
  set('d-V',     1331);
  set('d-rho',   d.rho_O);
  set('d-HO',    d.H_O);
  set('d-It',    d.I_now); set('d-MO', d.M_O);
  set('d-AO',    d.A_O);
  const surge = Math.abs(d.A_O||0) > 0.01;
  const sel = document.getElementById('d-surge');
  if(sel){sel.textContent=surge?'⚡ Belief surge detected':'No surge';sel.className=surge?'lv':'dim';}
  set('d-Cn', d.C_n); set('d-Vn', d.V_n); set('d-Gn', d.G_n); set('d-Qn', d.Q_n);
  const qel = document.getElementById('d-Qst');
  if(qel){qel.textContent=d.Q_accepted?'✓ Accepted (Q > τ)':'✗ Rejected (Q ≤ τ)';qel.className=d.Q_accepted?'ok':'fail';}
  set('d-IM', d.I_M); set('d-NM', d.N_M); set('d-CM', d.C_M); set('d-OM', d.O_M);

  // Live state
  set('d-ag',    d.step !== undefined ? `(${d.I_now})` : '—');
  set('d-goal',  '(9,9,9)');
  set('d-steps', d.step);
  set('d-vis',   d.visited_count);
  set('d-bel',   d.agent_belief);
  set('d-unc',   d.agent_unc);

  // Sparklines
  if(d.I_history){ hI.splice(0,hI.length,...d.I_history); spark('sp-I',hI,'#00ff88'); }
  hH.push(d.H_O||0); if(hH.length>80) hH.shift(); spark('sp-H',hH,'#44aaff');

  // patch live agent display properly
  set('d-ag', `(${d.step})`); // fallback — real agent pos comes from status
}

// get real agent pos from status poll
const _origPoll = pollStatus;
async function pollStatus2() {
  try {
    const s = await fetch('/api/status').then(r=>r.json());
    if (s.ready) {
      set('d-ag',   `(${s.agent.join(',')})`);
      set('d-goal', `(${s.goal.join(',')})`);
      set('d-steps', s.steps);
      if (s.reached) showReached(s.steps);
      if (!s.auto && autoOn) {
        autoOn=false;
        document.getElementById('abtn').textContent='▶ Auto Solve';
      }
    }
    const d = await fetch('/api/deep').then(r=>r.json());
    updateMath(d);
  } catch(e){}
}
clearInterval(pollTimer);
pollTimer = setInterval(pollStatus2, parseInt(document.getElementById('poll').value));

// ── Init ──────────────────────────────────────────────────
doReset();
</script>
</body>
</html>
"""

# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"""
╔═══════════════════════════════════════════════════════╗
║   Deep Observation 3D Cube Maze — Flask Public Server ║
║                                                       ║
║   http://localhost:{PORT}                              ║
║   http://0.0.0.0:{PORT}  (LAN / public)               ║
║                                                       ║
║   Stream:  /stream  (MJPEG, 24 fps)                   ║
║   API:     /api/reset  /api/step  /api/auto/start     ║
║            /api/record/start  /api/record/stop        ║
║                                                       ║
║   Ctrl+C to stop                                      ║
╚═══════════════════════════════════════════════════════╝
""")
    _new_maze()          # pre-generate so stream has something on first connect
    app.run(host='0.0.0.0', port=PORT, threaded=True)
