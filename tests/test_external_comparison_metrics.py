import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.paired_ccbs import DIAMETER, corridors, execute, score
from scripts.report_matched_actor_critic import paired, report


class CoordinationMetrics(unittest.TestCase):
    def test_crossing_between_ticks(self):
        a = np.array([[0., -1, 0], [2., 1, 0]])
        b = np.array([[0., 0, -1], [2., 0, 1]])
        m = score([a,b])
        self.assertAlmostEqual(m["min_separation"],0.)
        self.assertAlmostEqual(m["spacing_pair_s"], 2*DIAMETER/np.sqrt(2))
        self.assertFalse(m["safe"])

    def test_parked_goal_counts_as_occupied(self):
        a = np.array([[0.,0,0], [1.,1,0]])
        b = np.array([[0.,1,2], [3.,1,-1]])
        self.assertAlmostEqual(score([a,b])["spacing_pair_s"],2*DIAMETER)

    def test_frozen_conflict_strata(self):
        for seed in range(20):
            for high in (True, False):
                routes = corridors(seed, high)
                bare, status = execute(routes,False)
                self.assertEqual(status,"completed")
                self.assertEqual(score(bare)["safe"],not high)
                tpg, status = execute(routes,True)
                self.assertIn(status,("completed","horizon"))
                self.assertTrue(np.isfinite(score(tpg)["spacing_pair_s"]))


class NavigationMetrics(unittest.TestCase):
    def test_ties_and_discordant_pairs(self):
        def rows(seq):
            return [dict(ok=x) for x in seq]
        s = paired(rows([True,True,False]),rows([True,False,True]),"ok")
        self.assertEqual((s["ippo_wins"],s["ties"],s["ippo_losses"]), (1,1,1))
        self.assertEqual(s["exact_mcnemar_p"],1)
        self.assertEqual(paired(rows([True]),rows([True]),"ok")["difference"],0)

    def test_mismatched_targets_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = dict(manifest_index=0, scenario_hash="a", accepted_seed=41,
                           realized_scenario_hash="b", navigation_targets=[0,1,2], integration_failure=False,
                           strict_success=True, navigation_success=True)
            entry = dict(manifest_index=0, row={"scenario_hash":"a"}, pursuit_accepted_seed=41,
                         pursuit_realized_scenario_hash="b", pursuit_navigation_targets=[0,1,2])
            for label in ("train180","holdout100"):
                name = f"paired_{label}.json"
                (root/name).write_text(json.dumps(dict(rows=[entry],manifest_sha256="abc")))
                for arm in ("ppo","a2c","mappo"):
                    (root/arm).mkdir(exist_ok=True)
                    payload = dict(overall={"complete":True},manifest_sha256="abc",episodes=[example.copy()])
                    if arm == "a2c":
                        payload["episodes"][0]["navigation_targets"] = [2,1,0]
                    (root/arm/name).write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,"scenario mismatch"):
                report(root,root)


if __name__ == "__main__":
    unittest.main()
