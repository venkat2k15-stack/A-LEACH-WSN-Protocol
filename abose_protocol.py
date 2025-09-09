# abose_protocol.py - The original Abose et al. (2024) IMP-RES-EL protocol.

import math, random
import pandas as pd
import numpy as np

# --- Standard Parameters ---
N_NODES = 100
AREA_SIDE = 100.0
BS_POS = (AREA_SIDE/2, AREA_SIDE/2)
PACKET_SIZE = 4000
E_ELEC = 50e-9
E_DA = 0.5e-9 # Corrected value
E_FS = 10e-12
E_MP = 0.0013e-12
DO = math.sqrt(E_FS / E_MP) if E_MP > 0 else 87.0
INITIAL_ENERGY = 0.5
P_OPT = 0.05
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

class Node:
    def __init__(self, nid, x, y, energy=INITIAL_ENERGY):
        self.id = nid; self.x = x; self.y = y; self.energy = energy
        self.is_alive = True; self.is_CH = False; self.cluster = None

def tx_energy(bits, dist):
    if dist <= DO: return bits * (E_ELEC + E_FS * (dist**2))
    else: return bits * (E_ELEC + E_MP * (dist**4))

def rx_energy(bits): return bits * E_ELEC

def create_nodes(n=N_NODES):
    return [Node(i, random.uniform(0, AREA_SIDE), random.uniform(0, AREA_SIDE)) for i in range(n)]

def compute_threshold(node, round_num, alive_nodes, p=P_OPT):
    if not alive_nodes: return 0
    E_total = sum(n.energy for n in alive_nodes)
    E_avg = E_total / len(alive_nodes)
    Pi = p * (node.energy / E_avg) if E_avg > 0 else p
    Pi = max(min(Pi, 0.5), 0.001)
    try:
        threshold = Pi / (1 - Pi * (round_num % int(1.0 / Pi)))
    except (ValueError, ZeroDivisionError):
        threshold = Pi
    return threshold

def run_abose_simulation(rounds=2000):
    nodes = create_nodes(N_NODES)
    residual_energy = []; alive_nodes_count = []
    
    for r in range(1, rounds + 1):
        CHs = []
        alive_nodes = [n for n in nodes if n.is_alive]
        for node in alive_nodes:
            node.is_CH = False
            T = compute_threshold(node, r, alive_nodes)
            if random.random() <= T:
                node.is_CH = True; CHs.append(node)
        
        if not CHs and alive_nodes:
            best_node = max(alive_nodes, key=lambda n: n.energy)
            best_node.is_CH = True; CHs.append(best_node)

        for node in alive_nodes:
            if not node.is_CH and CHs:
                node.cluster = min(CHs, key=lambda ch: math.hypot(node.x - ch.x, node.y - ch.y))

        for node in alive_nodes:
            if not node.is_CH and node.cluster:
                dist = math.hypot(node.x - node.cluster.x, node.y - node.cluster.y)
                node.energy -= tx_energy(PACKET_SIZE, dist)
                node.cluster.energy -= rx_energy(PACKET_SIZE)
                if node.energy <= 0: node.is_alive = False

        for ch in CHs:
            if not ch.is_alive: continue
            members_count = sum(1 for n in alive_nodes if n.cluster == ch)
            ch.energy -= members_count * PACKET_SIZE * E_DA
            dist_bs = math.hypot(ch.x - BS_POS[0], ch.y - BS_POS[1])
            ch.energy -= tx_energy(PACKET_SIZE * (members_count + 1), dist_bs)
            if ch.energy <= 0: ch.is_alive = False

        alive_nodes_count.append(sum(1 for n in nodes if n.is_alive))
        residual_energy.append(sum(n.energy for n in nodes if n.is_alive))
        if sum(1 for n in nodes if n.is_alive) == 0:
            break

    # Pad results if simulation ends early
    while len(alive_nodes_count) < rounds:
        alive_nodes_count.append(0)
        residual_energy.append(0)

    return pd.DataFrame({'round': range(1, rounds + 1), 'alive_nodes': alive_nodes_count}), \
           pd.DataFrame({'round': range(1, rounds + 1), 'residual_energy': residual_energy})
