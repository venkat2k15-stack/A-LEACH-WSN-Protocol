# plot_results.py
# Loads the CSV results and generates comparative plots.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
RESULTS_DIR = Path("simulation_results_final")
PROTOCOLS = {
    "CS-Abose": "cs_abose_alive.csv",
    "MRP-GTCO": "mrpgtco_alive.csv",
    "Abose (Original)": "abose_alive.csv",
    "RLBEEP": "rlbeep_alive.csv",
    "EERPMS": "eerpms_alive.csv", # <-- ADDED
    "Sector-Based": "sector_alive.csv"
}
COLORS = {
    "CS-Abose": "red",
    "MRP-GTCO": "blue",
    "Abose (Original)": "green",
    "EERPMS": "magenta", # <-- ADDED
    "RLBEEP": "purple",
    "Sector-Based": "orange"
}
LINESTYLES = {
    "CS-Abose": "-",
    "MRP-GTCO": "--",
    "Abose (Original)": "-.",
    "EERPMS": "-", # <-- ADDED
    "RLBEEP": ":",
    "Sector-Based": "--"
}

def plot_alive_nodes():
    plt.figure(figsize=(12, 8))
    
    # Sort protocols to plot the proposed method last (on top)
    protocol_items = sorted(PROTOCOLS.items(), key=lambda item: item[0] != 'CS-Abose')

    for name, file in protocol_items:
        try:
            df = pd.read_csv(RESULTS_DIR / file)
            linewidth = 3 if name == "CS-Abose" else 2
            plt.plot(df['round'], df['alive_nodes'], label=name, 
                     color=COLORS.get(name), linestyle=LINESTYLES.get(name), linewidth=linewidth)
        except FileNotFoundError:
            print(f"Warning: Results file for {name} ('{file}') not found. Skipping.")

    plt.title("Network Lifetime Comparison: Alive Nodes vs. Rounds", fontsize=16, weight='bold')
    plt.xlabel("Simulation Rounds", fontsize=12)
    plt.ylabel("Number of Alive Nodes", fontsize=12)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.xlim(left=0, right=2000)
    plt.ylim(bottom=0, top=105)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_lifetime.png", dpi=300)
    print(f"Plot saved to {RESULTS_DIR / 'comparison_lifetime.png'}")
    plt.show()

if __name__ == "__main__":
    plot_alive_nodes()
