"""Fresh EXP27 point-target training after reference/compiled/GPU checks."""
from scripts import launch_mca_in_vitro as launcher


def main():
    root=launcher.ROOT
    launcher.PROTOCOL=root/'configs/experiments/EXP_0027_MCA_POINT_PURE_RL.json'
    launcher.OUT=root/'research/runs/EXP_0027_MCA_POINTS_20260930a'
    launcher.PREFLIGHT=root/'research/validation/EXP0027_GPU_PREFLIGHT_20260930'
    launcher.main()


if __name__=='__main__':main()
