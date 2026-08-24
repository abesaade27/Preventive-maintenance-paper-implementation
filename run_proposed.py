"""
run_proposed.py

Runs ONLY the proposed Change-Point-Gated Adaptive Memory scheme.

Workflow:
    ScenarioAgent
        ->
    ChangePointAgent
        ->
    Adaptive Memory Buffer
        ->
    FittingAgent
        ->
    OptimizerAgent
        ->
    ExplainerAgent

This script does NOT modify or compare against the paper baseline.

Outputs are saved under:
    output/proposed/

The original run.py outputs remain untouched.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import Simulation


# ---------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------

OUTPUT_DIR = "output/proposed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Run proposed scheme
# ---------------------------------------------------------------------

def run_proposed():

    print("=" * 70)
    print("PROPOSED SCHEME")
    print("Change-Point-Gated Adaptive Memory")
    print("=" * 70)

    sim = Simulation(
        n_conversations=10,

        # Same basic experimental scenario used by the paper
        Tp_init=25.0,

        # Cost settings
        Cp_start=20.0,
        Cp_step=25.5,
        Cc=250.0,

        # Scenario size
        n_units=50,

        # Regime change
        phase_switch_conv=6,
        beta_phase1=2.0,
        beta_phase2=0.6,
        eta_true=20.0,

        # Reproducible experiment
        seed=1,

        # Proposed mechanism
        use_change_point=True,

        # Change-point parameters
        alpha=0.01,
        min_buffer_batches=1,

        # None = unlimited adaptive memory
        max_buffer_size=None,

        # Reset stale memory when a regime change is detected
        reset_on_change=True,
    )

    results = sim.run(verbose=False)

    print()
    print("PROPOSED SCHEME RESULTS")
    print("-" * 70)

    for r in results:

        cp = r.get("change_point", {})

        print(f"Conversation {r['conversation_id']}")
        print(f"  Phase: {r['phase']}")
        print(f"  Beta: {r['beta_hat']:.3f}")
        print(f"  Eta: {r['eta_hat']:.3f}")
        print(f"  Buffer size: {r.get('buffer_size', 'N/A')}")
        print(f"  Buffer batches: {r.get('buffer_batches', 'N/A')}")
        print(f"  Change detected: {r.get('change_detected', False)}")
        print(f"  LR statistic: {r.get('change_point_lr', 0.0):.3f}")
        print(
            f"  Critical value: "
            f"{r.get('change_point_critical', 0.0):.3f}"
        )
        print(f"  Regime ID: {r.get('regime_id', 'N/A')}")

        if "dominant_policy" in r:
            print(f"  Dominant policy: {r['dominant_policy']}")

        if "Tp_star" in r:
            print(f"  Tp*: {r['Tp_star']:.3f}")

        if "CTE_star" in r:
            print(f"  CTE*: {r['CTE_star']:.3f}")

        print("-" * 70)

    return sim, results


# ---------------------------------------------------------------------
# Save raw results
# ---------------------------------------------------------------------

def save_results(sim, results):

    output_file = os.path.join(
        OUTPUT_DIR,
        "proposed_results.json"
    )

    with open(output_file, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=str
        )

    event_file = os.path.join(
        OUTPUT_DIR,
        "proposed_event_log.json"
    )

    with open(event_file, "w") as f:
        json.dump(
            [m.to_json() for m in sim.broker.event_log],
            f,
            indent=2,
            default=str
        )

    print()
    print(f"Saved: {output_file}")
    print(f"Saved: {event_file}")


# ---------------------------------------------------------------------
# Figure 1: Reliability re-estimation
# ---------------------------------------------------------------------

def plot_reliability(results):

    conv = [r["conversation_id"] for r in results]

    beta_hat = [r["beta_hat"] for r in results]
    beta_true = [r["beta_true"] for r in results]

    eta_hat = [r["eta_hat"] for r in results]
    eta_true = [r["eta_true"] for r in results]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5)
    )

    # Beta
    axes[0].plot(
        conv,
        beta_true,
        "--",
        label="True beta"
    )

    axes[0].plot(
        conv,
        beta_hat,
        "o-",
        label="Estimated beta"
    )

    axes[0].axvline(
        5.5,
        linestyle=":"
    )

    axes[0].set_xlabel("Conversation")
    axes[0].set_ylabel("Weibull beta")
    axes[0].set_title(
        "Proposed Scheme: Reliability Re-estimation"
    )

    axes[0].legend()

    # Eta
    axes[1].plot(
        conv,
        eta_true,
        "--",
        label="True eta"
    )

    axes[1].plot(
        conv,
        eta_hat,
        "o-",
        label="Estimated eta"
    )

    axes[1].axvline(
        5.5,
        linestyle=":"
    )

    axes[1].set_xlabel("Conversation")
    axes[1].set_ylabel("Weibull eta")
    axes[1].set_title(
        "Scale Parameter Adaptation"
    )

    axes[1].legend()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "fig10_proposed_reliability.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Figure 2: Change-point detection
# ---------------------------------------------------------------------

def plot_change_detection(results):

    conv = [r["conversation_id"] for r in results]

    lr = [
        r.get("change_point_lr", 0.0)
        for r in results
    ]

    critical = [
        r.get("change_point_critical", 0.0)
        for r in results
    ]

    detected = [
        r.get("change_detected", False)
        for r in results
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        conv,
        lr,
        "o-",
        label="Likelihood-ratio statistic"
    )

    ax.plot(
        conv,
        critical,
        "--",
        label="Critical threshold"
    )

    for x, y, d in zip(conv, lr, detected):

        if d:
            ax.scatter(
                x,
                y,
                s=100,
                marker="*",
                label="Change detected"
            )

    ax.axvline(
        5.5,
        linestyle=":"
    )

    ax.set_xlabel("Conversation")
    ax.set_ylabel("Likelihood-ratio statistic")
    ax.set_title(
        "Proposed Scheme: Change-Point Detection"
    )

    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()

    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys()
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "fig11_change_point_detection.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Figure 3: Adaptive memory
# ---------------------------------------------------------------------

def plot_adaptive_memory(results):

    conv = [r["conversation_id"] for r in results]

    buffer_size = [
        r.get("buffer_size", 0)
        for r in results
    ]

    buffer_batches = [
        r.get("buffer_batches", 0)
        for r in results
    ]

    detected = [
        r.get("change_detected", False)
        for r in results
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        conv,
        buffer_size,
        "o-",
        label="Adaptive buffer size"
    )

    for x, y, d in zip(
        conv,
        buffer_size,
        detected
    ):

        if d:
            ax.scatter(
                x,
                y,
                s=120,
                marker="*",
                label="Buffer reset"
            )

    ax.axvline(
        5.5,
        linestyle=":"
    )

    ax.set_xlabel("Conversation")
    ax.set_ylabel("Observations in memory")
    ax.set_title(
        "Proposed Scheme: Adaptive Memory Buffer"
    )

    handles, labels = ax.get_legend_handles_labels()

    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys()
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "fig12_adaptive_memory.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Figure 4: Buffer batches and regime
# ---------------------------------------------------------------------

def plot_regime_memory(results):

    conv = [r["conversation_id"] for r in results]

    batches = [
        r.get("buffer_batches", 0)
        for r in results
    ]

    regimes = [
        r.get("regime_id", 0)
        for r in results
    ]

    fig, ax1 = plt.subplots(
        figsize=(8, 4.5)
    )

    ax1.plot(
        conv,
        batches,
        "o-",
        label="Accumulated batches"
    )

    ax1.set_xlabel("Conversation")
    ax1.set_ylabel("Batches in current regime")

    ax1.axvline(
        5.5,
        linestyle=":"
    )

    ax2 = ax1.twinx()

    ax2.step(
        conv,
        regimes,
        where="mid",
        linestyle="--",
        label="Regime ID"
    )

    ax2.set_ylabel("Regime ID")

    ax1.set_title(
        "Proposed Scheme: Memory Growth and Regime Tracking"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "fig13_regime_memory.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Figure 5: Policy evolution
# ---------------------------------------------------------------------

def plot_policy(results):

    conv = [r["conversation_id"] for r in results]

    policies = [
        r.get(
            "dominant_policy",
            "unknown"
        )
        for r in results
    ]

    y = []

    for p in policies:

        if p == "oper_time_based":
            y.append(1)

        elif p == "calendar_based":
            y.append(0)

        else:
            y.append(np.nan)

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.step(
        conv,
        y,
        where="mid"
    )

    ax.axvline(
        5.5,
        linestyle=":"
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_yticklabels(
        [
            "Calendar-based",
            "Operating-time-based"
        ]
    )

    ax.set_xlabel("Conversation")

    ax.set_title(
        "Proposed Scheme: Preventive Maintenance Policy Evolution"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "fig14_proposed_policy.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Figure 6: Optimization
# ---------------------------------------------------------------------

def plot_optimization(results):

    conv = [r["conversation_id"] for r in results]

    tp_star = [
        r.get("Tp_star", np.nan)
        for r in results
    ]

    cte_star = [
        r.get("CTE_star", np.nan)
        for r in results
    ]

    fig, ax1 = plt.subplots(
        figsize=(8, 4.5)
    )

    ax1.plot(
        conv,
        tp_star,
        "o-",
        label="Optimal preventive interval Tp*"
    )

    ax1.axvline(
        5.5,
        linestyle=":"
    )

    ax1.set_xlabel("Conversation")
    ax1.set_ylabel("Optimal preventive interval")

    ax2 = ax1.twinx()

    ax2.plot(
        conv,
        cte_star,
        "s--",
        label="CTE*"
    )

    ax2.set_ylabel("Cost-time efficiency")

    ax1.set_title(
        "Proposed Scheme: Adaptive Maintenance Optimization"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "fig15_proposed_optimization.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_summary(results):

    detections = [
        r for r in results
        if r.get("change_detected", False)
    ]

    print()
    print("=" * 70)
    print("PROPOSED SCHEME SUMMARY")
    print("=" * 70)

    print(
        f"Total conversations: {len(results)}"
    )

    print(
        f"Change points detected: {len(detections)}"
    )

    if detections:

        print(
            "Detected conversation(s): "
            + ", ".join(
                str(r["conversation_id"])
                for r in detections
            )
        )

    final = results[-1]

    print(
        f"Final beta estimate: "
        f"{final['beta_hat']:.3f}"
    )

    print(
        f"Final eta estimate: "
        f"{final['eta_hat']:.3f}"
    )

    print(
        f"Final buffer size: "
        f"{final.get('buffer_size', 'N/A')}"
    )

    print(
        f"Final regime ID: "
        f"{final.get('regime_id', 'N/A')}"
    )

    print(
        f"Final policy: "
        f"{final.get('dominant_policy', 'N/A')}"
    )

    print("=" * 70)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    sim, results = run_proposed()

    save_results(
        sim,
        results
    )

    plot_reliability(
        results
    )

    plot_change_detection(
        results
    )

    plot_adaptive_memory(
        results
    )

    plot_regime_memory(
        results
    )

    plot_policy(
        results
    )

    plot_optimization(
        results
    )

    print_summary(
        results
    )

    print()
    print("=" * 70)
    print("ALL PROPOSED-SCHEME FIGURES GENERATED")
    print("=" * 70)


if __name__ == "__main__":
    main()
