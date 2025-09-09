# eerpms_protocol.py - Implementation of Yao et al. "EERPMS"

import math, random
import pandas as pd
import numpy as np
from collections import defaultdict

# --- Standard Parameters ---
N_NODES = 100
AREA_RADIUS = 100.0 # EERPMS paper uses a circular area
BS_POS = (0, 0) # Assumes BS is at the center (0,0) for angular calculations
PACKET_SIZE = 4000
E_ELEC = 50e-9
E_DA = 0.5e-9
E_FS = 10e-12
E_MP = 0.0013e-12
DO = math.sqrt(E_FS / E_MP) if E_MP > 0 else 87.0
INITIAL_ENERGY = 0.5
SEED = 42

# Protocol-specific parameters for CH selection (Attribute Function)
W1_ENERGY = 0.7 # w1 in the paper
W2_LOCATION = 0.3 # w2 in the paper

random.seed(SEED)
np.random.seed(SEED)

class Node:
    def __init__(self, nid, x, y, energy=INITIAL_ENERGY):
        self.id = nid; self.x = x; self.y = y; self.energy = energy
        self.is_alive = True; self.is_CH = False; self.cluster_id = -1

def tx_energy(bits, dist):
    if dist <= DO: return bits * (E_ELEC + E_FS * (dist**2))
    else: return bits * (E_ELEC + E_MP * (dist**4))

def rx_energy(bits): return bits * E_ELEC

def create_nodes(n=N_NODES):
    # Create nodes within a circular area
    nodes = []
    for i in range(n):
        r = AREA_RADIUS * math.sqrt(random.random())
        theta = random.random() * 2 * math.pi
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        nodes.append(Node(i, x + BS_POS[0], y + BS_POS[1]))
    return nodes

def run_eerpms_simulation(rounds=2000):
    nodes = create_nodes(N_NODES)
    alive_nodes_count = []
    
    for r in range(1, rounds + 1):
        alive_nodes = [n for n in nodes if n.is_alive]
        num_alive = len(alive_nodes)
        if num_alive == 0: break

        # --- Centralized Calculation at BS ---
        # 1. Calculate optimal K* and d_CHtoBS (Eqs. 15 & 16 in paper)
        try:
            k_opt = int(round(( (3/4) * (math.pi**2) * num_alive )**(1/3.0)))
            k_opt = max(1, k_opt) # Ensure at least one cluster
            d_opt = math.sqrt((2 * num_alive * AREA_RADIUS**2) / (3 * (num_alive + k_opt)))
        except (ValueError, ZeroDivisionError):
            k_opt = 1; d_opt = AREA_RADIUS / 2

        # 2. Cluster Formation based on Angle (simplified Otsu)
        clusters = defaultdict(list)
        sector_angle = 360 / k_opt
        for node in alive_nodes:
            angle = math.degrees(math.atan2(node.y - BS_POS[1], node.x - BS_POS[0]))
            if angle < 0: angle += 360
            node.cluster_id = int(angle / sector_angle)
            clusters[node.cluster_id].append(node)

        # 3. CH Selection in each cluster
        CHs = []
        for cid, cluster_nodes in clusters.items():
            if not cluster_nodes: continue

            # Calculate distances to optimal circle for normalization
            dist_to_opt_circle = [abs(math.hypot(n.x - BS_POS[0], n.y - BS_POS[1]) - d_opt) for n in cluster_nodes]
            d_max = max(dist_to_opt_circle); d_min = min(dist_to_opt_circle)
            
            best_node_for_ch = None
            max_score = -float('inf')

            for i, node in enumerate(cluster_nodes):
                # Attribute Function F2(Si) (Eq. 34 in paper)
                energy_term = W1_ENERGY * (node.energy / INITIAL_ENERGY)
                
                # Location term requires normalization within the cluster
                dist_val = dist_to_opt_circle[i]
                if (d_max - d_min) > 1e-6:
                    location_term = W2_LOCATION * ((d_max - dist_val) / (d_max - d_min))
                else:
                    location_term = W2_LOCATION # All are equidistant
                
                score = energy_term + location_term
                if score > max_score:
                    max_score = score
                    best_node_for_ch = node
            
            if best_node_for_ch:
                best_node_for_ch.is_CH = True
                CHs.append(best_node_for_ch)
        
        # --- Data Transmission and Energy Dissipation ---
        for node in alive_nodes:
            if not node.is_CH:
                # Find its CH
                my_ch = next((ch for ch in CHs if ch.cluster_id == node.cluster_id), None)
                if my_ch and my_ch.is_alive:
                    dist = math.hypot(node.x - my_ch.x, node.y - my_ch.y)
                    node.energy -= tx_energy(PACKET_SIZE, dist)
                    my_ch.energy -= rx_energy(PACKET_SIZE)
                    if node.energy <= 0: node.is_alive = False
        
        for ch in CHs:
            if not ch.is_alive: continue
            members_count = sum(1 for n in alive_nodes if not n.is_CH and n.cluster_id == ch.cluster_id)
            ch.energy -= members_count * PACKET_SIZE * E_DA
            dist_bs = math.hypot(ch.x - BS_POS[0], ch.y - BS_POS[1])
            ch.energy -= tx_energy(PACKET_SIZE * (members_count + 1), dist_bs)
            if ch.energy <= 0: ch.is_alive = False
            
        # Reset CH status for next round
        for node in nodes: node.is_CH = False

        alive_nodes_count.append(sum(1 for n in nodes if n.is_alive))

    # Pad results if simulation ends early
    while len(alive_nodes_count) < rounds: alive_nodes_count.append(0)
        
    return pd.DataFrame({'round': range(1, rounds + 1), 'alive_nodes': alive_nodes_count}), None
