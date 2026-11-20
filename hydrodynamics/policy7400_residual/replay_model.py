"""Run the water-force model on a prepared trajectory bundle or its packaged fixture."""
import argparse
from pathlib import Path
import numpy as np
from final_runtime import WaterForces

INPUTS=('hull_motion','oar_motion','hull_map','oar_map','previous_physics','friction','quasi_local','oar_rotation')

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('input',nargs='?',type=Path,default=Path(__file__).with_name('example.npz'))
    cli.add_argument('--output',type=Path)
    args=cli.parse_args()
    model=WaterForces()
    with np.load(args.input) as archive:
        arrays=[archive[k] for k in INPUTS]
        if len({len(a) for a in arrays})!=1 or len(arrays[0])==0:
            raise ValueError('All eight inputs must cover the same nonempty trajectory')
        parts=np.array([model.components(*row) for row in zip(*arrays)])
        wrench=parts.sum(axis=1)
        if 'expected' in archive:
            np.testing.assert_allclose(wrench,archive['expected'],rtol=.00002,atol=.0008)
    if args.output:
        np.savez_compressed(args.output,wrench=wrench,physics=parts[:,0],residual=parts[:,1])
    print(f'{len(wrench)} frames, shape {wrench.shape}, finite={np.isfinite(wrench).all()}')

if __name__=='__main__':
    main()
