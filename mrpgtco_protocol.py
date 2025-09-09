# mrpgtco_protocol.py - FINAL CORRECTED Implementation of Yao et al. "MRP-GTCO"

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
COMM_RADIUS = 40  # For neighbor discovery in game theory phase
CH_COVERAGE_RADIUS = 45 # For final CH selection coverage
W_ALPHA = 0.7  # Weight for energy in penalty function
W_BETA = 0.3   # Weight for node degree in penalty function

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

def tx_energy(bits, dist):
    if dist <= DO: return bits * (E_ELEC + E_FS * (dist**2))
    else: return bits * (E_ELEC + E_MP * (dist**4))

def rx_energy(bits): return bits * E_ELEC

def create_nodes(n=N_NODES):
    return [Node(i, random.uniform(0, AREA_SIDE), random.uniform(0, AREA_SIDE)) for i in range(n)]

def run_mrpgtco_simulation(rounds=2000):
    nodes = create_nodes(N_NODES)
    alive_nodes_count = []
    
    for r in range(1, rounds + 1):
        alive_nodes = [n for n in nodes if n.is_alive]
        if not alive_nodes: break

        for node in nodes:
            node.is_CH = False
            node.cluster = None

        # --- Stage 1: Candidate CH Selection (Game Theory) ---
        candidate_chs = []
        for node in alive_nodes:
            neighbors = [n for n in alive_nodes if n.id != node.id and math.hypot(node.x - n.x, node.y - n.y) < COMM_RADIUS]
            if not neighbors: continue
            
            er_max = max(n.energy for n in neighbors)
            er_min = min(n.energy for n in neighbors)
            er_diff = er_max - er_min if (er_max - er_min) > 0 else 1.0
            
            # Probability is higher for nodes with more energy
            p_ch = (node.energy - er_min) / er_diff
            if random.random() < p_ch:
                candidate_chs.append(node)

        # --- Stage 2: Final CH Selection (Coverage & Load Balancing) ---
        final_chs = []
        uncovered_nodes = list(alive_nodes)
        
        while uncovered_nodes and candidate_chs:
            best_candidate = None
            max_coverage = -1
            
            # Find the candidate that covers the most uncovered nodes
            for cand in candidate_chs:
                covered_count = sum(1 for n in uncovered_nodes if math.hypot(n.x - cand.x, n.y - cand.y) <= CH_COVERAGE_RADIUS)
                if covered_count > max_coverage:
                    max_coverage = covered_count
                    best_candidate = cand
            
            if best_candidate:
                final_chs.append(best_candidate)
                candidate_chs.remove(best_candidate)
                # Update the list of uncovered nodes
                uncovered_nodes = [n for n in uncovered_nodes if math.hypot(n.x - best_candidate.x, n.y - best_candidate.y) > CH_COVERAGE_RADIUS]
            else:
                # No candidate can cover any more nodes, so break
                break
        
        if not final_chs and alive_nodes:
            final_chs.append(max(alive_nodes, key=lambda n: n.energy))

        for ch in final_chs:
            ch.is_CH = True
        
        # --- Data Transmission Phase ---
        
        # 1. Non-CH nodes transmit to their nearest CH
        ch_data_load = defaultdict(lambda: 1) # Each CH starts with its own packet
        for node in alive_nodes:
            if not node.is_CH and final_chs:
                node.cluster = min(final_chs, key=lambda ch: math.hypot(node.x - ch.x, node.y - ch.y))
                if node.cluster and node.cluster.is_alive:
                    dist = math.hypot(node.x - node.cluster.x, node.y - node.cluster.y)
                    if node.energy > tx_energy(PACKET_SIZE, dist):
                        node.energy -= tx_energy(PACKET_SIZE, dist)
                        node.cluster.energy -= rx_energy(PACKET_SIZE)
                        ch_data_load[node.cluster.id] += 1

        # 2. CHs aggregate data
        for ch in final_chs:
            if not ch.is_alive: continue
            num_packets_aggregated = ch_data_load[ch.id] - 1
            if num_packets_aggregated > 0:
                ch.energy -= num_packets_aggregated * PACKET_SIZE * E_DA

        # 3. CHs transmit to BS (with multi-hop relay logic)
        sorted_chs = sorted(final_chs, key=lambda c: math.hypot(c.x-BS_POS[0], c.y-BS_POS[1]), reverse=True)

        for ch in sorted_chs:
            if not ch.is_alive: continue
            
            total_bits = ch_data_load[ch.id] * PACKET_SIZE
            
            dist_direct_to_bs = math.hypot(ch.x - BS_POS[0], ch.y - BS_POS[1])
            cost_direct = tx_energy(total_bits, dist_direct_to_bs)

            best_relay = None
            min_cost = cost_direct
            
            # Find the best relay (another CH that is closer to the BS)
            for relay in final_chs:
                if relay.id == ch.id or not relay.is_alive: continue
                
                dist_ch_to_relay = math.hypot(ch.x - relay.x, ch.y - relay.y)
                dist_relay_to_bs = math.hypot(relay.x - BS_POS[0], relay.y - BS_POS[1])
                
                if dist_relay_to_bs < dist_direct_to_bs and dist_ch_to_relay < DO * 2: # Relay must be closer
                    cost_via_relay = tx_energy(total_bits, dist_ch_to_relay)
                    if cost_via_relay < min_cost:
                        min_cost = cost_via_relay
                        best_relay = relay
            
            # Execute transmission
            if best_relay:
                # Transmit to relay
                ch.energy -= tx_energy(total_bits, math.hypot(ch.x - best_relay.x, ch.y - best_relay.y))
                # The relay's load for the *next* round would increase, but for this round,
                # we assume it aggregates and forwards immediately.
                best_relay.energy -= rx_energy(total_bits)
                best_relay.energy -= tx_energy(total_bits, math.hypot(best_relay.x - BS_POS[0], best_relay.y - BS_POS[1]))
            else:
                # Transmit directly
                ch.energy -= cost_direct

        # Update node status
        for node in nodes:
            if node.energy <= 0:
                node.is_alive = False

        alive_nodes_count.append(sum(1 for n in nodes if n.is_alive))

    while len(alive_nodes_count) < rounds:
        alive_nodes_count.append(0)
        
    return pd.DataFrame({'round': range(1, rounds + 1), 'alive_nodes': alive_nodes_count}), None
