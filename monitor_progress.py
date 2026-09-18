#!/usr/bin/env python3
"""实时监控自动科研系统的进度"""

import json
import time
from pathlib import Path

def monitor():
    results_dir = Path("auto_research_results")
    history_file = results_dir / "experiment_history.json"

    print("=" * 80)
    print("🔍 自动科研监控")
    print("=" * 80)
    print()

    while True:
        try:
            if history_file.exists():
                with open(history_file) as f:
                    data = json.load(f)

                experiments = data.get("experiments", [])
                baseline = data.get("baseline")

                print(f"\r已完成实验: {len(experiments)}/4 | ", end="")

                if experiments:
                    latest = experiments[-1]
                    name = latest["config"]["name"]
                    success = latest["final_success_rate"]
                    removal = latest["final_removal_rate"]
                    duration = latest["duration_seconds"] / 60

                    print(f"最新: {name} | 成功率: {success:.1%} | 溶解率: {removal:.1%} | 用时: {duration:.1f}分钟", end="")
                else:
                    print("等待第一个实验完成...", end="")
            else:
                print(f"\r等待自动科研系统启动...", end="")

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n\n监控已停止")
            break
        except Exception as e:
            print(f"\r错误: {e}", end="")
            time.sleep(5)

if __name__ == "__main__":
    monitor()
