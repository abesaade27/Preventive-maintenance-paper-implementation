from simulation import Simulation

sim = Simulation(
    n_conversations=10,
    n_units=50,

    phase_switch_conv=6,

    beta_phase1=2.0,
    beta_phase2=0.6,
    eta_true=20.0,

    Cp_start=20.0,
    Cp_step=25.5,
    Cc=250.0,

    seed=42,

    # Proposed Change-Point + Adaptive Memory scheme
    use_change_point=True,

    alpha=0.01,
    min_buffer_batches=1,
    max_buffer_size=None,
    reset_on_change=True
)

results = sim.run()

print("\n===== PROPOSED SCHEME RESULTS =====\n")

for r in results:
    cp = r.get("change_point", {})

    print(f"Conversation {r['conversation_id']}")

    if r.get("beta_hat") is not None:
        print(f"  Beta: {r['beta_hat']:.3f}")

    if r.get("eta_hat") is not None:
        print(f"  Eta: {r['eta_hat']:.3f}")

    if cp:
        print(f"  Change detected: {cp['triggered']}")
        print(f"  Buffer size: {cp['buffer_size_after']}")
        print(f"  LR statistic: {cp['lr_stat']:.3f}")
        print(f"  Regime ID: {cp['regime_id']}")

    print("-" * 40)
