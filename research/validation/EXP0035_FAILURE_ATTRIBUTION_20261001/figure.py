"""Summary figure for FAILURE_ANALYSIS_EXP35.md (read-only)."""
import json, sys
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = sys.argv[1]
rf = json.load(open(f'{D}/route_following.json'))['table']; iv = json.load(open(f'{D}/interventions.json'))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
labels = ['[0,.12)', '[.12,.5)', '[.5,1)', '[1,2)', '[2,4)', '[4,8)', '[8,16)', '16+']
keys = ['d[0,0.12)', 'd[0.12,0.5)', 'd[0.5,1)', 'd[1,2)', 'd[2,4)', 'd[4,8)', 'd[8,16)', 'd[16,1000000000.0)']
for al, ls in (('alive1', '-'), ('alive4', ':')):
    for o, c in (('S', '#2b7bba'), ('F', '#d9472b')):
        ax[0].plot(labels, [rf[f'{al}|{k}|{o}']['prog_mm_s'] for k in keys], ls, color=c, marker='o', label=f'{al[-1]} target(s) left, {"success" if o == "S" else "failure"}')
ax[0].axhline(1, color='gray', lw=.8); ax[0].set_ylim(-.3, 1.05)
ax[0].set_title('Progress toward own target\n(robot speed 1 mm/s)', fontsize=10); ax[0].set_xlabel('geodesic distance to own target (mm)'); ax[0].set_ylabel('mm/s'); ax[0].legend(fontsize=8); ax[0].tick_params(axis='x', rotation=45)
modes = ['policy', 'near_map', 'last_map', 'far_map', 'gate0.3']
base = {42: .51, 43: .78, 44: .65}
vals = [[base[s] for s in (42, 43, 44)]] + [[o['success'] for o in iv[m]] for m in modes[1:]]
x = np.arange(len(modes))
for j, s in enumerate((42, 43, 44)):
    ax[1].bar(x+(j-1)*.25, [100*v[j] for v in vals], .25, label=f'seed {s}')
ax[1].axhline(80, color='k', ls='--', lw=.8); ax[1].set_xticks(x); ax[1].set_xticklabels(['policy\n(replay)', 'map\n<=0.7mm', 'map\nlast tgt', 'map\n>0.7mm', 'stop\n|a|<0.3'], fontsize=8)
ax[1].set_ylabel('complete clearance (%)'); ax[1].set_title('Eval-only substitutions, 100 layouts x 3 seeds'); ax[1].legend(fontsize=8)
tx = json.load(open(f'{D}/taxonomy.json'))['ALL']
names = list(tx); ax[2].barh(range(len(names)), [tx[n] for n in names], color='#777')
ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels([n.split(' (')[0] for n in names], fontsize=8); ax[2].invert_yaxis()
ax[2].set_title(f'Failure taxonomy (n={sum(tx.values())})'); ax[2].set_xlabel('failed episodes')
fig.tight_layout(); fig.savefig(f'{D}/summary.png', dpi=130)
