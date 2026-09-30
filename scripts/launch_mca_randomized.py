"""EXP28: continue compatible pure RL weights on validated random domains."""
from scripts import launch_mca_in_vitro as launcher


def main():
    root=launcher.ROOT
    launcher.PROTOCOL=root/'configs/experiments/EXP_0028_MCA_RANDOMIZED_PURE_RL.json'
    launcher.OUT=root/'research/runs/EXP_0028_MCA_RANDOMIZED_20260930a'
    launcher.PREFLIGHT=root/'research/validation/EXP0028_GPU_PREFLIGHT_20260930'
    launcher.main()


if __name__=='__main__':main()
