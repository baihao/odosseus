"""Exp21 frozen evaluation on val (last two laps, t = 897..1120 s; map = training run t < 897 s)."""
import sys; sys.path.insert(0, '.')
from odo import online
out = 'outputs/21_online/val'
pred, inf, ms, stat = online.run('val', out)
m = online.evaluate(out, 'val', 'Exp21 online localiser: car-state EKF + map PnP (7 cams, real time)', ms, stat, video=True,
                    baselines={'Exp12 PnP (no filter)': 'outputs/12_reloc_pnp/val/metrics_val.json',
                               'Exp01 dead reckoning': 'outputs/01_deadreckoning/metrics_val.json'},
                    others={'Exp12 PnP': 'outputs/12_reloc_pnp/val/pred.csv'})
print('VAL', {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, stat, f'{ms:.1f} ms/frame')
