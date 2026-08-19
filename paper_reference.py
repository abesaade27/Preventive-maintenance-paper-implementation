"""
paper_reference.py
Ground-truth numbers transcribed directly from Table 1 and Table 2 of the
published paper (Crespo Marquez & Gomez Fernandez, 2026, Expert Systems
With Applications 314, 131767). Used only for side-by-side comparison in
run.py -- the paper's own random draws can't be reproduced (no published
seed), but the *inputs* (Tp_init, Cp schedule, Cc, phase switch, n_units)
are matched exactly so the qualitative behavior of our run can be checked
against the qualitative behavior of the published run.
"""

PAPER_TABLE1 = [
    # conv, phase,   beta_true, eta_true, beta_hat, eta_hat, Cp,   Cc,   Tp_used, Tp_cal,  CTE_cal, Tp_op,  CTE_op, dominant_policy
    (1,  "Phase1",  2.000, 20.000, 1.772, 18.356, 20.0,  250.0, 25.000,   5.107,  8.992,   5.416,  8.655, "oper_time_based"),
    (2,  "Phase1",  2.000, 20.000, 2.564, 14.262, 45.6,  250.0,  5.416,   6.167, 12.111,   6.722, 11.334, "oper_time_based"),
    (3,  "Phase1",  2.000, 20.000, 1.224, 42.450, 71.1,  250.0,  6.722,  51.653,  7.530, 102.446,  6.281, "oper_time_based"),
    (4,  "Phase1",  2.000, 20.000, 1.881, 20.043, 96.7,  250.0,102.446,  12.938, 15.955,  17.971, 13.070, "oper_time_based"),
    (5,  "Phase1",  2.000, 20.000, 2.404, 19.730,122.2,  250.0, 17.971,  12.722, 16.451,  17.545, 13.202, "oper_time_based"),
    (6,  "Phase 2", 0.600, 20.000, 0.426, 34.666,147.8,  250.0, 17.545, 173.332,  3.718, 173.332,  4.940, "calendar_based"),
    (7,  "Phase 2", 0.600, 20.000, 0.632, 31.782,173.3,  250.0,173.332, 158.909,  5.444, 158.909,  6.448, "calendar_based"),
    (8,  "Phase 2", 0.600, 20.000, 0.605, 17.903,198.9,  250.0,158.909,  89.517,  9.614,  89.517, 11.296, "calendar_based"),
    (9,  "Phase 2", 0.600, 20.000, 0.642, 21.360,224.4,  250.0, 89.517, 106.802,  8.683, 106.802,  9.783, "calendar_based"),
    (10, "Phase 2", 0.600, 20.000, 0.718, 14.297,250.0,  250.0,106.802,  71.484, 14.603,  71.484, 15.406, "calendar_based"),
]
PAPER_TABLE1_COLUMNS = ["conv", "phase", "beta_true", "eta_true", "beta_hat", "eta_hat",
                         "Cp", "Cc", "Tp_used", "Tp_cal", "CTE_cal", "Tp_op", "CTE_op", "dominant_policy"]

PAPER_TABLE2 = {
    "Phase 1": {"beta_hat_mean": 1.969, "beta_hat_std": 0.535, "eta_hat_mean": 22.968, "eta_hat_std": 11.131,
                "Tp_star_mean": 30.020, "Tp_star_std": 40.910, "CTE_star_mean": 10.508, "CTE_star_std": 2.992,
                "censoring_mean": 0.492, "censoring_std": 0.415},
    "Phase 2": {"beta_hat_mean": 0.605, "beta_hat_std": 0.108, "eta_hat_mean": 24.002, "eta_hat_std": 8.840,
                "Tp_star_mean": 120.009, "Tp_star_std": 44.202, "CTE_star_mean": 8.412, "CTE_star_std": 4.204,
                "censoring_mean": 0.124, "censoring_std": 0.200},
}

# Inputs confirmed directly from the paper's Fig. 1/Fig. 4 tick-log example
# and Table 1's own Cp/Tp_used columns.
PAPER_INPUTS = dict(
    Tp_init=25.0,
    Cp_schedule=[20.0, 45.6, 71.1, 96.7, 122.2, 147.8, 173.3, 198.9, 224.4, 250.0],
    Cc=250.0,
    n_units=50,          # Fig. 4's tick-log example: "Number of observations: 50, censored: 8"
    beta_phase1=2.0,
    beta_phase2=0.6,
    eta_true=20.0,
    phase_switch_conv=6,
    n_conversations=10,
)
