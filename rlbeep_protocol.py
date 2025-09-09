# rlbeep_protocol.py - CORRECTED Implementation of Abadi et al. "RLBEEP"

import math, random
import pandas as pd
import numpy as np
from collections import defaultdict

# --- Standard Parameters ---
N_NODES = 100
AREA_SIDE = 100.0
BS_POS = (AREA_SIDE/2, AREA_SIDE/2)
PACKET_SIZE = 4000
E_ELEC = 50e-9
E_DA = 0.5e-9
E_FS = 10e-12
E_MP = 0.0013e-12
DO = math.sqrt(E_FS / E_MP)
INITIAL_ENERGY = 0.5
SEED = 42

# Protocol-specific parameters
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.5
EPSILON = 0.2
P_OPT = 0.1 # Use a higher CH probability for stability

random.seed(SEED)
np.random.seed(SEED)

class Node:
    def __init__(self, nid, x, y, energy=INITIAL_ENERGY):
        self.id = nid; self.x = x; self.y = y; self.energy = energy
        self.is_alive = True; self.is_CH = False; self.cluster = None
        self.q_table = defaultdict(float)

def tx_energy(bits, dist):
    if dist <= DO: return bits * (E_ELEC + E_FS * (dist**2))
    else: return bits * (E_ELEC + E_MP * (dist**4))

def rx_energy(bits): return bits * E_ELEC

def create_nodes(n=N_NODES):
    return [Node(i, random.uniform(0, AREA_SIDE), random.uniform(0, AREA_SIDE)) for i in range(n)]

def calculate_reward(neighbor_node):
    dist_to_bs = math.hypot(neighbor_node.x - BS_POS[0], neighbor_node.y - BS_POS[1])
    return neighbor_node.energy / (dist_to_bs + 1e-6)

def run_rlbeep_simulation(rounds=2000):
    nodes = create_nodes(N_NODES)
    alive_nodes_count = []
    
    for r in range(1, rounds + 1):
        alive_nodes = [n for n in nodes if n.is_alive]
        if not alive_nodes: break

        # 1. Clustering Phase (essential for fair comparison)
        for node in nodes: node.is_CH = False; node.cluster = None
        
        CHs = []
        for node in alive_nodes:
            if random.random() < P_OPT:
                node.is_CH = True
                CHs.append(node)
        
        if not CHs and alive_nodes:
            best_node = max(alive_nodes, key=lambda n: n.energy)
            best_node.is_CH = True; CHs.append(best_node)

        for node in alive_nodes:
            if not node.is_CH and CHs:
                node.cluster = min(CHs, key=lambda ch: math.hypot(node.x - ch.x, node.y - ch.y))

        # 2. Data transmission from non-CH to CH
        ch_data_load = defaultdict(int)
        for node in alive_nodes:
            if not node.is_CH and node.cluster and node.cluster.is_alive:
                dist = math.hypot(node.x - node.cluster.x, node.y - node.cluster.y)
                node.energy -= tx_energy(PACKET_SIZE, dist)
                node.cluster.energy -= rx_energy(PACKET_SIZE)
                ch_data_load[node.cluster.id] += 1

        # 3. CHs use RL to route data to BS
        for ch in CHs:
            if not ch.is_alive: continue
            
            total_bits = (ch_data_load[ch.id] + 1) * PACKET_SIZE
            ch.energy -= ch_data_load[ch.id] * PACKET_SIZE * E_DA

            current_ch = ch
            while current_ch.is_alive and math.hypot(current_ch.x - BS_POS[0], current_ch.y - BS_POS[1]) > DO:
                # Neighbors are other CHs
                alive_ch_neighbors = [c for c in CHs if c.is_alive and c.id != current_ch.id]
                if not alive_ch_neighbors: break

                next_hop = None
                if random.random() < EPSILON:
                    next_hop = random.choice(alive_ch_neighbors)
                else:
                    best_q = -float('inf')
                    for neighbor in alive_ch_neighbors:
                        q_val = current_ch.q_table.get(neighbor.id, 0)
                        if q_val > best_q: best_q = q_val; next_hop = neighbor
                    if not next_hop: next_hop = random.choice(alive_ch_neighbors)
                
                dist = math.hypot(current_ch.x - next_hop.x, current_ch.y - next_hop.y)
                current_ch.energy -= tx_energy(total_bits, dist)
                next_hop.energy -= rx_energy(total_bits)
                
                reward = calculate_reward(next_hop)
                max_q_next = max([next_hop.q_table.get(c.id,0) for c in alive_ch_neighbors if c.id != next_hop.id] or [0])
                old_q = current_ch.q_table.get(next_hop.id, 0)
                new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_q_next - old_q)
                current_ch.q_table[next_hop.id] = new_q
                
                current_ch = next_hop

            if current_ch.is_alive:
                dist_bs = math.hypot(current_ch.x - BS_POS[0], current_ch.y - BS_POS[1])
                current_ch.energy -= tx_energy(total_bits, dist_bs)

        for node in nodes:
            if node.energy <= 0: node.is_alive = False

        alive_nodes_count.append(sum(1 for n in nodes if n.is_alive))
        
    while len(alive_nodes_count) < rounds: alive_nodes_count.append(0)
    return pd.DataFrame({'round': range(1, rounds + 1), 'alive_nodes': alive_nodes_count}), None
