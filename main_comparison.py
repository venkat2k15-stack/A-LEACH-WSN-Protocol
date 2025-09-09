
from pathlib import Path

# Import all protocol runners
from abose_protocol import run_abose_simulation
from cs_abose_protocol import run_cs_abose_simulation
from sector_protocol import run_sector_simulation
from rlbeep_protocol import run_rlbeep_simulation
from mrpgtco_protocol import run_mrpgtco_simulation
from eerpms_protocol import run_eerpms_simulation

# --- Simulation Parameters ---
ROUNDS = 2000
OUTPUT_DIR = Path("simulation_results_final")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("--- Starting Full Comparative Simulation ---")
    
    # Run Proposed Method
    print("1/6: Running CS-Abose (Proposed)...")
    # FIX: CS-Abose now returns 2 items, not 3
    df_alive_cs, df_energy_cs = run_cs_abose_simulation(rounds=ROUNDS)
    df_alive_cs.to_csv(OUTPUT_DIR / "cs_abose_alive.csv", index=False)
    print("   ...CS-Abose complete.")
    
    # Run the baselines
    print("2/6: Running Abose (Original)...")
    df_alive_abose, _ = run_abose_simulation(rounds=ROUNDS)
    df_alive_abose.to_csv(OUTPUT_DIR / "abose_alive.csv", index=False)
    print("   ...Abose complete.")
    
    print("3/6: Running Sector-Based...")
    df_alive_sector, _ = run_sector_simulation(rounds=ROUNDS)
    df_alive_sector.to_csv(OUTPUT_DIR / "sector_alive.csv", index=False)
    print("   ...Sector-Based complete.")
    
    print("4/6: Running RLBEEP...")
    df_alive_rlbeep, _ = run_rlbeep_simulation(rounds=ROUNDS)
    df_alive_rlbeep.to_csv(OUTPUT_DIR / "rlbeep_alive.csv", index=False)
    print("   ...RLBEEP complete.")
    
    print("5/6: Running MRP-GTCO...")
    df_alive_mrpgtco, _ = run_mrpgtco_simulation(rounds=ROUNDS)
    df_alive_mrpgtco.to_csv(OUTPUT_DIR / "mrpgtco_alive.csv", index=False)
    print("   ...MRP-GTCO complete.")
    
    print("6/6: Running EERPMS...")
    df_alive_eerpms, _ = run_eerpms_simulation(rounds=ROUNDS)
    df_alive_eerpms.to_csv(OUTPUT_DIR / "eerpms_alive.csv", index=False)
    print("   ...EERPMS complete.")

    print("\n--- All simulations complete! ---")
    print(f"Results are saved in the '{OUTPUT_DIR}' folder.")
    print("You can now run plot_results.py to visualize the comparison.")

if __name__ == "__main__":
    main()

