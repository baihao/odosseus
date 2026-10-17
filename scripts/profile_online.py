"""Profile one online run with cProfile (sorted by cumulative time)."""
import cProfile, pstats, sys, json
sys.path.insert(0, '.')
from odo import online
cams = sys.argv[1].split(',') if len(sys.argv) > 1 else ['fr']
pr = cProfile.Profile(); pr.enable()
online.run('tune', 'outputs/21_online/prof', cams=cams, limit=150, verbose=False, cfg=dict(extra_dof=3))
pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(25)
