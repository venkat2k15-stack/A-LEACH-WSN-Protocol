# cs_abose_protocol.py -- CS-Abose: A Compressive-Sensing-Aware Protocol based on IMP-RES-EL
# Built upon the Abose et al. (2024) protocol.
# FINAL VERSION: Removed faulty sklearn dependency to focus on energy modeling.

import math, random
import pandas as pd
import numpy as np
from pathlib import Path

# --- Parameters (Matched with baselines) ---
N_NODES = 100
AREA_SIDE = 100.0
BS_POS = (AREA_SIDE/2, AREA_SIDE/2)
PACKET_SIZE = 4000 # Bits per reading before compression
E_ELEC = 50e-9
E_DA = 0.5e-9 # Corrected value as per our previous discussion
E_FS = 10e-12
E_MP = 0.0013e-12
DO = math.sqrt(E_FS / E_MP) if E_MP > 0 else 87.0
INITIAL_ENERGY = 0.5
P_OPT = 0.05
SEED = 42

### NEW: Compressive Sensing Parameters ###
CS_RATIO = 0.25 # Compress data to 25% of its original size
# The number of bits for a compressed packet. Assume each measurement is a float (64 bits).
# The total bits sent will depend on the number of members in the cluster.
BITS_PER_MEASUREMENT = 64 

random.seed(SEED)
np.random.seed(SEED)

class Node:
    def __init__(self, nid, x, y, energy=INITIAL_ENERGY):
        self.id = nid
        self.x = x
        self.y = y
        self.energy = energy
        self.is_alive = True
        self.is_CH = False
        self.cluster = None
        self.times_as_CH = 0

    def distance_to(self, pos):
        return math.hypot(self.x - pos[0], self.y - pos[1])

def tx_energy(bits, dist):
    if dist <= DO:
        return bits * (E_ELEC + E_FS * (dist**2))
    else:
        return bits * (E_ELEC + E_MP * (dist**4))

def rx_energy(bits):
    return bits * E_ELEC

def create_nodes(n=N_NODES):
    return [Node(i, random.uniform(0, AREA_SIDE), random.uniform(0, AREA_SIDE)) for i in range(n)]

def compute_cs_aware_threshold(node, round_num, nodes, p=P_OPT, w_energy=0.7, w_data=0.3):
    alive_nodes = [n for n in nodes if n.is_alive]
    if not alive_nodes: return 0
    E_total = sum(n.energy for n in alive_nodes)
    E_avg = E_total / len(alive_nodes)
    energy_factor = p * (node.energy / E_avg) if E_avg > 0 else p
    
    dist_from_center = node.distance_to(BS_POS)
    max_dist = math.hypot(AREA_SIDE/2, AREA_SIDE/2)
    coverage_factor = (1 - dist_from_center / max_dist) * p * 2
    
    Pi_cs = (w_energy * energy_factor) + (w_data * coverage_factor)
    Pi_cs = max(min(Pi_cs, 0.5), 0.001)

    try:
        threshold = Pi_cs / (1 - Pi_cs * (round_num % int(1.0 / Pi_cs)))
    except (ValueError, ZeroDivisionError):
        threshold = Pi_cs
    return threshold

def run_cs_abose_simulation(rounds=2000):
    nodes = create_nodes(N_NODES)
    residual_energy = []
    alive_nodes_count = []
    
    for r in range(1, rounds + 1):
        alive_nodes = [n for n in nodes if n.is_alive]
        if not alive_nodes: break

        CHs = []
        for node in alive_nodes:
            node.is_CH = False
            node.cluster = None
            T = compute_cs_aware_threshold(node, r, nodes)
            if random.random() <= T:
                node.is_CH = True
                CHs.append(node)

        if not CHs and alive_nodes:
            best_node = max(alive_nodes, key=lambda n: n.energy)
            best_node.is_CH = True
            CHs.append(best_node)

        for node in alive_nodes:
            if not node.is_CH and CHs:
                node.cluster = min(CHs, key=lambda ch: node.distance_to((ch.x, ch.y)))
            
        for node in alive_nodes:
            if not node.is_CH and node.cluster and node.cluster.is_alive:
                dist = node.distance_to((node.cluster.x, node.cluster.y))
                node.energy -= tx_energy(PACKET_SIZE, dist)
                node.cluster.energy -= rx_energy(PACKET_SIZE)
                if node.energy <= 0: node.is_alive = False

        for ch in CHs:
            if not ch.is_alive: continue
            
            cluster_members = [n for n in alive_nodes if n.cluster == ch]
            num_members = len(cluster_members)
            
            # --- CORRECTED ENERGY MODEL FOR COMPRESSION ---
            if num_members > 0:
                # Energy for data aggregation for all member packets
                ch.energy -= num_members * PACKET_SIZE * E_DA
                
                # Calculate the size of the compressed packet
                # n_components = number of compressed measurements
                n_components = int(num_members * CS_RATIO)
                
                # If compression is not possible (e.g., only 1 member), send uncompressed
                if n_components == 0 and num_members > 0:
                    total_bits_transmitted = num_members * PACKET_SIZE
                else:
                    total_bits_transmitted = n_components * BITS_PER_MEASUREMENT

                # Energy for transmitting the final packet (either compressed or not) to BS
                dist_bs = ch.distance_to(BS_POS)
                ch.energy -= tx_energy(total_bits_transmitted, dist_bs)
            
            if ch.energy <= 0: ch.is_alive = False

        alive_nodes_count.append(sum(1 for n in nodes if n.is_alive))
        residual_energy.append(sum(n.energy for n in nodes if n.is_alive))

    # Pad results if simulation ends early
    while len(alive_nodes_count) < rounds:
        alive_nodes_count.append(0)
        residual_energy.append(0)
        
    # We now only return the two essential dataframes
    return pd.DataFrame({'round': range(1, rounds + 1), 'alive_nodes': alive_nodes_count}), \
           pd.DataFrame({'round': range(1, rounds + 1), 'residual_energy': residual_energy})

