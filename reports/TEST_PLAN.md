# Test Plan

| Test | Purpose | Command | Expected Result | Status |
|------|---------|---------|----------------|--------|
| Split sanity | No train/test overlap | `python3 scripts/check_splits.py --splits configs/splits.yaml` | 4 safe, 4 leaky identified | PASS |
| BCFF tautology | Document vacuous objective | `python3 tests/test_bcff_tautology.py` | coeffs ≈ [1,1,0,0] across seeds | PASS |
| Conflict: identical adapters | High conflict for identical | `python3 tests/test_conflict_diagnostics.py` | cosine > 0.99 | PASS |
| Conflict: orthogonal adapters | Low conflict for orthogonal | same | cosine < 0.15 | PASS |
| Conflict: activation-dependent | Same-U different-V differs | same | energy/cosine differs under different H | PASS |
| CARR: base equivalence | Zero residuals → output = input | `python3 tests/test_conflict_aware_routing.py` | diff < 1e-5 | PASS |
| CARR: valid gate distribution | Gates ≥ 0 and sum to 1 | same | (2,5,5) shape, sum ≈ 1 | PASS |
| CARR: no reliability mode | Works without reliability | same | runs without error | PASS |
| CARR: no conflict mode | Works without conflict | same | runs without error | PASS |
| CARR: gate stats logged | Diagnostics returned | same | stats dict with expected keys | PASS |
| Smoke test: all imports | New modules importable | `python3 -c "from src.conflict_diagnostics import *; from src.conflict_aware_routing import *"` | No import error | NOT RUN |
| One-batch overfit | Router learns on toy data | Needs pod / GPU | Loss decreases, gates change | NOT RUN |
| Checkpoint integrity | Base gate=1 reproduces base | Needs pod / GPU | Exact match | NOT RUN |
| A/B/C comparison | Full CARR > static > no-mechanism | Needs pod / GPU | Minimal evidence | NOT RUN |
