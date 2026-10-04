"""16 colour/shape stats (robust_common.hand_features, unchanged) for every crop in the v2 training cache.
Used for (a) the colour-rule-vs-model comparison on held-out sources and (b) the photo-level hand-Mahalanobis gate in
the deploy-rule simulation. Writes CACHE/hand.npy (N,16) float32, same row order as CACHE/meta.json."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import train_common as tc

_C = None
def _init():
    global _C
    _C = np.load(tc.CACHE / "crops.npy", mmap_mode="r")

def _feat(rng):
    import robust_common as rc
    return np.stack([rc.hand_features(np.asarray(_C[i])) for i in range(*rng)])

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    n = len(np.load(tc.CACHE / "crops.npy", mmap_mode="r")); step = 2000
    t0 = time.time()
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 10, initializer=_init) as ex:
        out = list(ex.map(_feat, [(a, min(n, a + step)) for a in range(0, n, step)]))
    H = np.concatenate(out).astype(np.float32)
    np.save(tc.CACHE / "hand.npy", H)
    tc.log(f"hand features {H.shape} in {time.time()-t0:.0f}s; NaN rows {int(np.isnan(H).any(1).sum())}")
