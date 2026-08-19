# check_cpa.py — save inside pm_agentic/ folder, run with: python check_cpa.py
from simulation import Simulation

sim = Simulation(n_units=50, seed=1, alpha=0.01, min_buffer_batches=1)
results = sim.run(verbose=False)
for r, log in zip(results, sim.change_point.change_log):
    print(r["conversation_id"], r["beta_hat"], log["buffer_batches_before"], log["triggered"], log["lr_stat"])