import numpy as np
config = {
    "max_factor": 300,
    "exp_type": "create_beta_func_fig_1",
    "num_seeds": 200,
    "target_var_factor": 10,
    "dimension": 2,
    "source_var_range": [1e-1, 1],
    "source_var_equal": 0.1,
    "beta_bar": 0.9,
    "delta": 0.05,
    "equal_source_variance": True,
    "r_bar_func": "log_T",
    "constant_t": 100,
    "constant_n": int(1e5),
    "factor_q_with_target_error": True,
    "different_factors": {
        "close": 0,
        "middle": 10,
        "far": int(2e4)
    },
    "kappa": 1,
    "close_ratio": 0.1

}
