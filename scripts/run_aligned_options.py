"""EXP0052 uses the tested SMDP PPO runner with explicit registered factories."""
import argparse
import sys
from scripts import run_option_learning as runner
from scripts.aligned_option_episode import AlignedOptionEpisode,PROTOCOL,aligned_hashes
from marl.aligned_option_learning import AllocationPriorActorCritic


def main():
    extra=argparse.ArgumentParser(add_help=False)
    extra.add_argument('--decision-steps',type=int)
    selected,remaining=extra.parse_known_args()
    if selected.decision_steps is not None:
        if 'train' in remaining or ('--policy' in remaining and remaining[remaining.index('--policy')+1]=='learned'):
            raise ValueError('Decision override is only for separately registered conventional/untrained arms')
        if selected.decision_steps<1:raise ValueError('Positive decision interval required')
    def episode_factory(*args,**kwargs):
        if selected.decision_steps is not None:kwargs['option_steps']=selected.decision_steps
        return AlignedOptionEpisode(*args,**kwargs)
    runner.OptionEpisode=episode_factory
    runner.OptionActorCritic=AllocationPriorActorCritic
    runner.PROTOCOL=PROTOCOL
    runner.option_hashes=aligned_hashes
    sys.argv=[sys.argv[0]]+remaining
    runner.main()


if __name__=='__main__':main()
