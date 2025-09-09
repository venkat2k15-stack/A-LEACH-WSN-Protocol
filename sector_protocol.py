import math
import random
import pandas as pd
import numpy as np
from collections import defaultdict

# --- Standard Parameters ---
N_NODES = 100
AREA_SIDE = 100.0
BS_POS = (AREA_SIDE / 2, AREA_SIDE / 2)
PACKET_SIZE = 4000
E_ELEC = 50e-9
E_DA = 0.5e-9
E_FS = 10e-12
E_MP = 0.0013e-12
DO = math.sqrt(E_FS / E_MP) if E_MP > 0 else 87.0
INITIAL_ENERGY = 0.5
SEED = 42

# Protocol-specific parameters
SECTOR_PERCENTAGE = 0.2  # p in the paper
NUM_SECTORS = int(N_NODES * SECTOR_PERCENTAGE)
SECTOR_ANGLE = 360 / NUM_SECTORS

# Weights for priority: alpha (distance), beta (energy), gamma (density)
W_DIST = 0.3
W_ENERGY = 0.4
W_DENSITY = 0.3

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
        self.sector = -1
        self.priority = 0.0

def tx_energy(bits, dist):
    if dist <= DO:
        return bits * (E_ELEC + E_FS * (dist ** 2))
    else:
        return bits * (E_ELEC + E_MP * (dist ** 4))

def rx_energy(bits):
    return bits * E_ELEC

def create_nodes(n=N_NODES):
    return [Node(i, random.uniform(0, AREA_SIDE), random.uniform(0, AREA_SIDE)) for i in range(n)]

def run_sector_simulation(rounds=2000):
    nodes = create_nodes(N_NODES)
    alive_nodes_count = []
    residual_energy = []

    for r in range(1, rounds + 1):
        sectors = defaultdict(list)

        for node in nodes:
            node.is_CH = False
            if not node.is_alive:
                continue

            # 1. Sectorization
            angle = math.degrees(math.atan2(node.y - BS_POS[1], node.x - BS_POS[0]))
            if angle < 0:
                angle += 360
            node.sector = int(angle / SECTOR_ANGLE)
            sectors[node.sector].append(node)

        CHs = []
        for sector_id, sector_nodes in sectors.items():
            if not sector_nodes:
                continue

            # 2. Priority Calculation
            for node in sector_nodes:
                dist_to_bs = math.hypot(node.x - BS_POS[0], node.y - BS_POS[1])
                node_density = len(sector_nodes)

                node.priority = (
                    W_ENERGY * (node.energy / INITIAL_ENERGY) +
                    W_DIST * (1 - dist_to_bs / (AREA_SIDE * 1.414)) +
                    W_DENSITY * (node_density / len(nodes))
                )

            # 3. CH Selection
            ch = max(sector_nodes, key=lambda n: n.priority)
            ch.is_CH = True
            CHs.append(ch)

        alive_nodes = [n for n in nodes if n.is_alive]
        for node in alive_nodes:
            if not node.is_CH:
                ch_in_sector = next((c for c in CHs if c.sector == node.sector), None)
                if ch_in_sector:
                    dist = math.hypot(node.x - ch_in_sector.x, node.y - ch_in_sector.y)
                    node.energy -= tx_energy(PACKET_SIZE, dist)
                    ch_in_sector.energy -= rx_energy(PACKET_SIZE)
                    if node.energy <= 0:
                        node.is_alive = False

        for ch in CHs:
            if not ch.is_alive:
                continue
            dist_bs = math.hypot(ch.x - BS_POS[0], ch.y - BS_POS[1])
            ch.energy -= tx_energy(PACKET_SIZE, dist_bs)
            if ch.energy <= 0:
                ch.is_alive = False

        alive_nodes_count.append(sum(1 for n in nodes if n.is_alive))
        if sum(1 for n in nodes if n.is_alive) == 0:
            break

    while len(alive_nodes_count) < rounds:
        alive_nodes_count.append(0)

    return pd.DataFrame({'round': range(1, rounds + 1), 'alive_nodes': alive_nodes_count}), None

