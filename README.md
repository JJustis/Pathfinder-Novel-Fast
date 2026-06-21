# Deep Observation Pathfinding: A Hierarchical Bayesian Framework for Informational Awareness

## 1. Introduction

The Deep Observation framework is a hierarchical Bayesian inference system designed for autonomous navigation under uncertainty. It enables an agent to build an accurate model of its environment by fusing noisy observations at multiple scales, propagating information both upward (from fine to coarse) and downward (from coarse to fine). When no clear path exists, the agent explores the most uncertain regions to maximize information gain.

This document presents the complete mathematical formulation of the system, from the observation model and Bayesian updates to the uncertainty-driven exploration rule.

---

## 2. The Observation Model – Multi-Scale Sensing

The agent observes its environment at three concentric radii, corresponding to three layers:

- **Layer 0 (innermost)**: radius \( R_0 = 2 \), lowest noise
- **Layer 1 (middle)**: radius \( R_1 = 4 \), moderate noise
- **Layer 2 (outermost)**: radius \( R_2 = 8 \), highest noise

Let the true occupancy of cell \((x,y)\) be:

\[
o(x,y) \in \{0,1\}, \quad 0 = \text{wall}, \quad 1 = \text{free}
\]

The observation at layer \( k \) for cell \((x,y)\) is:

\[
z^{(k)}_{x,y} \sim \mathcal{N}(o(x,y),\, \sigma_k^2)
\]

where:

\[
\sigma_0^2 = 0.05, \quad \sigma_1^2 = 0.10, \quad \sigma_2^2 = 0.20
\]

The measured value is clipped to \([0,1]\) to represent a probability:

\[
\tilde{z}^{(k)}_{x,y} = \max(0, \min(1, z^{(k)}_{x,y}))
\]

---

## 3. Hierarchical Belief Representation

Each layer maintains a belief map \( b^{(k)}_{x,y} \), representing the probability that cell \((x,y)\) is free:

\[
b^{(k)}_{x,y} = P(\text{cell }(x,y) \text{ is free} \mid \text{all observations up to time } t)
\]

Initially, all beliefs are set to 0.5 (complete ignorance):

\[
b^{(k)}_{x,y}(0) = 0.5 \quad \forall k, x, y
\]

---

## 4. Belief Update – The Upward Pass

The upward pass propagates information from the innermost layer (most precise) to the outermost layers (coarsest).

### 4.1 Layer 0 (Innermost)

The posterior belief for layer 0 is computed as a weighted average of the prior belief and the new observation:

\[
b^{(0)}_{\text{post}} = \frac{\alpha_0 \cdot \tilde{z}^{(0)} + \beta_0 \cdot b^{(0)}_{\text{prior}}}{\alpha_0 + \beta_0}
\]

where:

\[
\alpha_0 = \sigma_0^2, \quad \beta_0 = \sigma_0^2 \cdot 0.5
\]

### 4.2 Layers \( i > 0 \) (Outer Layers)

For each outer layer \( i \), the prior is taken from the posterior of the layer below:

\[
b^{(i)}_{\text{prior}} = b^{(i-1)}_{\text{post}}
\]

The posterior is then:

\[
b^{(i)}_{\text{post}} = \frac{\alpha_i \cdot \tilde{z}^{(i)} + \beta_i \cdot b^{(i)}_{\text{prior}}}{\alpha_i + \beta_i}
\]

where:

\[
\alpha_i = \sigma_i^2, \quad \beta_i = \sigma_{i-1}^2 \cdot 0.5
\]

This ensures that fine-scale information propagates upward, so that even the coarsest layer is informed by the detailed observations of the inner layers.

---

## 5. Belief Update – The Downward Pass

The downward pass propagates information from the outermost layers inward, refining the inner layers with global structure.

For \( i \) from \( N-2 \) down to \( 0 \):

\[
b^{(i)}_{\text{refined}} = \alpha \cdot b^{(i+1)}_{\text{post}} + (1 - \alpha) \cdot b^{(i)}_{\text{post}}
\]

where:

\[
\alpha = 0.3
\]

This is a convex combination: the refined belief is a weighted average of the outer layer's posterior and the current inner layer's posterior. The outer layer provides global context, while the inner layer retains local detail.

---

## 6. The Fused Belief Map

After one full cycle (upward + downward), the fused belief map is taken as the innermost layer's refined belief:

\[
b_{\text{fused}}(x,y) = b^{(0)}_{\text{refined}}(x,y)
\]

This map represents the agent's best estimate of the environment, combining high-resolution local data with global structural information.

---

## 7. Path Planning on the Fused Belief Map

The fused belief map is thresholded to produce an obstacle map:

\[
\text{obstacle}(x,y) =
\begin{cases}
1 & \text{if } b_{\text{fused}}(x,y) < \tau \\
0 & \text{otherwise}
\end{cases}
\]

where \( \tau = 0.45 \) is the threshold.

The agent uses A* to plan a path from its current position \( \mathbf{p} \) to the goal \( \mathbf{g} \). The cost function is:

\[
f(\mathbf{n}) = g(\mathbf{n}) + h(\mathbf{n})
\]

where:

- \( g(\mathbf{n}) \) is the cost from the start to node \( \mathbf{n} \) (Manhattan distance)
- \( h(\mathbf{n}) = |n_x - g_x| + |n_y - g_y| \) is the heuristic (Manhattan distance to goal)

If A* returns a path, the agent follows it step-by-step, re-planning after each move.

---

## 8. Uncertainty-Driven Exploration

If A* cannot find a path (i.e., the belief map is too uncertain to connect the agent to the goal), the agent switches to exploration mode.

The uncertainty of a cell is defined as the variance of a Bernoulli distribution:

\[
u(x,y) = b_{\text{fused}}(x,y) \cdot (1 - b_{\text{fused}}(x,y))
\]

This is maximised when \( b = 0.5 \) (maximum uncertainty) and minimised when \( b \to 0 \) or \( b \to 1 \) (certainty).

The agent selects the adjacent free cell with the highest uncertainty, preferring unvisited cells to avoid loops. The selection rule is:

\[
\mathbf{p}_{\text{next}} = \arg\max_{\mathbf{n} \in \mathcal{N}(\mathbf{p})} \left( u(\mathbf{n}) + \epsilon \right)
\]

where:

- \( \mathcal{N}(\mathbf{p}) \) is the set of free neighbours of current position \( \mathbf{p} \)
- \( \epsilon \sim \text{Uniform}(-0.01, 0.01) \) is a small random tie-breaker

If unvisited neighbours exist, the agent restricts the search to them:

\[
\mathcal{N}_{\text{unvisited}}(\mathbf{p}) = \{ \mathbf{n} \in \mathcal{N}(\mathbf{p}) \mid \mathbf{n} \notin \mathcal{V} \}
\]

where \( \mathcal{V} \) is the set of visited cells. If \( \mathcal{N}_{\text{unvisited}} \neq \emptyset \), the search is limited to it.

---

## 9. The Complete Cycle – Information Flow

The system operates as a closed loop. Each step enables the next:

1. **Observation**: The agent senses the environment at multiple radii (\( R_0, R_1, R_2 \)).

2. **Upward Pass**: Fine-scale information propagates outward, ensuring consistency across layers.

3. **Downward Pass**: Global structure refines local beliefs, resolving ambiguities.

4. **Path Planning**: A* attempts to find a path using the fused belief map.

5. **Exploration**: If no path exists, the agent moves to the most uncertain adjacent cell, maximising information gain.

6. **Movement**: The agent physically moves, changing its position and enabling new observations.

---

## 10. Connection to the Original Diagram

The concentric circles from the diagram correspond to the three layers:

- **Maximal Radius of Influence** = Layer 2 (outermost, \( R_2 = 8 \), highest noise)
- **Maximal Effect of Phenomenon** = Layer 1 (middle, \( R_1 = 4 \), moderate noise)
- **Maximum Effect of Observation** = Layer 0 (innermost, \( R_0 = 2 \), lowest noise)
- **Deep Observation (b)** = The fused belief map after upward and downward passes

The arrows in the diagram represent:

- **Upward arrows**: Information flow from fine to coarse (observations → outer layers)
- **Downward arrows**: Information flow from coarse to fine (outer → inner layers for refinement)

This bidirectional propagation is the core of the hierarchical Bayesian inference system.

---

## 11. Mathematical Summary

| Component | Formula |
|-----------|---------|
| **Observation** | \( z^{(k)}_{x,y} \sim \mathcal{N}(o(x,y), \sigma_k^2) \) |
| **Upward Pass (Layer 0)** | \( b^{(0)}_{\text{post}} = \frac{\alpha_0 \tilde{z}^{(0)} + \beta_0 b^{(0)}_{\text{prior}}}{\alpha_0 + \beta_0} \) |
| **Upward Pass (Layer i > 0)** | \( b^{(i)}_{\text{post}} = \frac{\alpha_i \tilde{z}^{(i)} + \beta_i b^{(i-1)}_{\text{post}}}{\alpha_i + \beta_i} \) |
| **Downward Pass** | \( b^{(i)}_{\text{refined}} = \alpha b^{(i+1)}_{\text{post}} + (1-\alpha) b^{(i)}_{\text{post}} \) |
| **Fused Map** | \( b_{\text{fused}} = b^{(0)}_{\text{refined}} \) |
| **Uncertainty** | \( u(x,y) = b_{\text{fused}}(x,y) \cdot (1 - b_{\text{fused}}(x,y)) \) |
| **Exploration** | \( \mathbf{p}_{\text{next}} = \arg\max_{\mathbf{n} \in \mathcal{N}} (u(\mathbf{n}) + \epsilon) \) |

---

## 12. Conclusion

The Deep Observation framework provides a mathematically rigorous approach to autonomous navigation under uncertainty. By fusing multi-scale observations, propagating information bidirectionally, and exploring uncertain regions, the agent builds an accurate belief map and reliably reaches its goal. The mathematics is transparent, the architecture is modular, and the principles extend to any domain where sensors provide noisy, multi-resolution data.

___________________________________________________________________________________________________________________________________________________
Below is a **block diagram** of the **Deep Observation Pathfinding** framework. It shows the main components and the flow of information between them, as described in the mathematical formulation.

---

## Block Diagram

```
+---------------------+
|   ENVIRONMENT       |
|  (True Occupancy)   |
+---------------------+
           |
           | noisy observations at radii R2,R1,R0
           v
+---------------------+     +---------------------+
|  OBSERVATION LAYER  |     |  OBSERVATION LAYER  |     +---------------------+
|     (Layer 0)       |     |     (Layer 1)       |     |  OBSERVATION LAYER  |
|   R=2, σ²=0.05      |     |   R=4, σ²=0.10     |     |     (Layer 2)       |
|   fine, precise     |     |   middle            |     |   R=8, σ²=0.20     |
+---------------------+     +---------------------+     |   coarse, noisy    |
           |                         |                         |                  |
           |                         |                         |                  |
           v                         v                         v                  |
+---------------------+     +---------------------+     +---------------------+  |
|  BELIEF MAP b^(0)   |     |  BELIEF MAP b^(1)   |     |  BELIEF MAP b^(2)   |  |
|   prior = 0.5       |     |   prior = 0.5       |     |   prior = 0.5       |  |
+---------------------+     +---------------------+     +---------------------+  |
           |                         |                         |                  |
           |                         |                         |                  |
           +------------+------------+-------------------------+                  |
                        |             |                                           |
                        |             |                                           |
                        v             v                                           |
           +---------------------------+                                         |
           |    UPWARD PASS            |                                         |
           |  (fine → coarse)          |                                         |
           |  b^(i)_post = ...         |                                         |
           +---------------------------+                                         |
                        |                                                       |
                        v                                                       |
           +---------------------------+                                         |
           |    DOWNWARD PASS          |                                         |
           |  (coarse → fine)          |                                         |
           |  b^(i)_refined = ...      |                                         |
           +---------------------------+                                         |
                        |                                                       |
                        |                                                       |
                        v                                                       |
           +---------------------------+                                         |
           |  FUSED BELIEF MAP         |                                         |
           |  b_fused = b^(0)_refined  |                                         |
           +---------------------------+                                         |
                        |                                                       |
            +-----------+----------+--------------------------------------------+
            |                      |
            v                      v
+---------------------------+   +---------------------------+
|   PATH PLANNING (A*)     |   |  UNCERTAINTY-DRIVEN      |
|   threshold τ = 0.45     |   |  EXPLORATION             |
|   cost: g + h            |   |  u = b(1-b)              |
+---------------------------+   |  p_next = argmax(u+ε)   |
            |                      +---------------------------+
            |                                  |
            |                                  |
            v                                  |
   +---------------------+                    |
   |   PATH FOUND?       |                    |
   |   (yes/no)          |                    |
   +---------------------+                    |
            |                                  |
       yes  |  no                              |
            v                                  |
   +---------------------+                    |
   |  Follow Path        |                    |
   |  (move along path)  |                    |
   +---------------------+                    |
            |                                  |
            |                                  |
            +------------+---------------------+
                         |
                         v
             +---------------------+
             |   AGENT MOVES       |
             |   new position      |
             +---------------------+
                         |
                         |
                         +------> back to ENVIRONMENT (new observations)
```

---

## Explanation of Each Block

### 1. Observation Layers
- The agent receives **noisy observations** at three scales:
  - **Layer 0** (innermost, radius 2, low noise σ²=0.05) – high precision.
  - **Layer 1** (middle, radius 4, σ²=0.10) – moderate.
  - **Layer 2** (outermost, radius 8, σ²=0.20) – noisy but broad.
- Observations are Gaussian samples clipped to [0,1], representing the probability that a cell is free.

### 2. Belief Maps
- Each layer maintains a **belief map** `b^(k)` – the probability that each cell is free.
- Initially all set to 0.5 (complete ignorance).

### 3. Upward Pass (Fine → Coarse)
- Information propagates from the most precise layer (0) outward.
- **Layer 0**: posterior = weighted average of prior and observation.
- **Layer i > 0**: prior is taken from the posterior of layer i‑1.
- This ensures that even the coarsest layer is informed by fine details.

### 4. Downward Pass (Coarse → Fine)
- Information flows back from the outermost layer inward.
- Each layer is refined by mixing with the layer above (α = 0.3).
- Global structure disambiguates local uncertainties.

### 5. Fused Belief Map
- After one full upward+downward cycle, the innermost layer becomes the **fused map**.
- This is the agent’s best estimate, combining local detail and global context.

### 6. Path Planning (A*)
- The fused map is thresholded (τ = 0.45) to produce an obstacle map.
- A* searches for the shortest path from agent to goal using Manhattan heuristic.
- If a path exists, the agent follows it step‑by‑step, re‑planning after each move.

### 7. Uncertainty‑Driven Exploration
- If A* fails (no path), the agent switches to exploration.
- **Uncertainty** `u(x,y) = b_fused * (1 - b_fused)` is maximised when belief is 0.5.
- The agent moves to the adjacent free cell with highest uncertainty, preferring unvisited cells.
- A small random tie‑breaker `ε` prevents deterministic loops.

### 8. Movement & Cycle
- After moving, the agent’s position changes.
- New observations are taken, and the cycle repeats.
- This closed‑loop process continues until the goal is reached.

---

## Information Flow Summary

| Direction      | Meaning |
|----------------|---------|
| **Upward**     | Detailed local observations → coarse global beliefs (informs outer layers) |
| **Downward**   | Global structure → refines local beliefs (resolves ambiguity) |
| **Planning**   | Uses fused map to guide movement toward goal |
| **Exploration**| When stuck, maximises information gain by visiting uncertain cells |

---

## Key Equations in the Diagram

- **Observation**:  
  \( z^{(k)}_{x,y} \sim \mathcal{N}(o(x,y), \sigma_k^2) \)

- **Upward Pass (Layer 0)**:  
  \( b^{(0)}_{\text{post}} = \frac{\alpha_0 \tilde{z}^{(0)} + \beta_0 b^{(0)}_{\text{prior}}}{\alpha_0 + \beta_0} \)

- **Upward Pass (Layer i > 0)**:  
  \( b^{(i)}_{\text{post}} = \frac{\alpha_i \tilde{z}^{(i)} + \beta_i b^{(i-1)}_{\text{post}}}{\alpha_i + \beta_i} \)

- **Downward Pass**:  
  \( b^{(i)}_{\text{refined}} = \alpha \, b^{(i+1)}_{\text{post}} + (1-\alpha) \, b^{(i)}_{\text{post}} \)

- **Uncertainty**:  
  \( u(x,y) = b_{\text{fused}}(x,y) \cdot (1 - b_{\text{fused}}(x,y)) \)

- **Exploration Selection**:  
  \( \mathbf{p}_{\text{next}} = \arg\max_{\mathbf{n} \in \mathcal{N}} \bigl( u(\mathbf{n}) + \epsilon \bigr) \)

This diagram and explanation capture the complete hierarchical Bayesian inference system described in the document.
