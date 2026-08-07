# pipeline.fetch pulls in histdata/tenacity — fetch-only dependencies not installed in
# every consumer of this package (e.g. the dashboard backend container only ever needs
# compute_features/normalize_features, via live_feature_bridge.py; found 2026-08-07 when
# that import chain broke the container with "No module named 'histdata'"). Guarded so a
# partial-dependency environment still gets everything else; fetch_all_assets is simply
# unavailable (None) there instead of the whole package failing to import.
try:
    from pipeline.fetch import fetch_all_assets
except ImportError:
    fetch_all_assets = None

from pipeline.features import compute_features
from pipeline.normalize import normalize_features
from pipeline.align import align_assets
from pipeline.validate import validate_pipeline

__all__ = [
    "fetch_all_assets",
    "compute_features",
    "normalize_features",
    "align_assets",
    "validate_pipeline",
]
