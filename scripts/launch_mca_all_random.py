"""EXP29: all-object randomization and explicit avoidance observations."""
from scripts import launch_mca_in_vitro as launcher


def main():
    root=launcher.ROOT
    launcher.PROTOCOL=root/'configs/experiments/EXP_0029_MCA_ALL_RANDOM_PURE_RL.json'
    launcher.OUT=root/'research/runs/EXP_0029_MCA_ALL_RANDOM_20260930a'
    launcher.PREFLIGHT=root/'research/validation/EXP0029_GPU_PREFLIGHT_20260930'
    launcher.main()


if __name__=='__main__':main()
