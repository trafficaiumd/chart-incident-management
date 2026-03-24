"""
Analyzer: load media, build 3×4 grid, call model, return normalized incident dict.

Use this from the dashboard "Run Analysis" flow.
"""

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_layer.loader import MediaLoader
from input_layer.grid import GridCreator


def _get_default_model():
    """Use model stub until real VLM (Gemini/GPT) is wired."""
    try:
        from tests.model_stubs import perfect_model
        return perfect_model
    except ImportError:
        # Fallback if tests not on path
        def _stub(path: str) -> Dict[str, Any]:
            return {
                "incident_type": "unknown",
                "confidence": 0.0,
                "vehicles": {"total_count": 0, "involved_in_incident": [], "nearby_traffic": [], "types": []},
                "lanes": {"total": 4, "blocked": [], "shoulder_blocked": False},
                "hazards": {"fire": False, "smoke": False, "debris": False, "injuries_visible": False, "spill": False},
                "traffic": {"state": "unknown", "queue_length_meters": 0},
                "emergency_response": {
                    "police_present": False, "officers_count": 0,
                    "ambulance_present": False, "fire_truck_present": False, "responders_on_scene": False,
                },
                "analysis_failed": False,
            }
        return _stub


def load_and_build_grid(
    file_path: str,
    location_text: str = "",
    max_frames: int = 12,
) -> Optional[Path]:
    """
    Load image or video from disk, build 3×4 grid, save to temp file.

    Args:
        file_path: Path to image (.jpg, .png) or video (.mp4, .avi, etc.)
        location_text: Optional overlay text (e.g. "I-95 NB @ MM 27")
        max_frames: Max frames for video sampling

    Returns:
        Path to temp grid image, or None on failure.
    """
    path = Path(file_path)
    if not path.exists():
        return None

    loader = MediaLoader()
    grid_creator = GridCreator(rows=3, cols=4, cell_size=(300, 300))

    suffix = path.suffix.lower()
    frames = []

    if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        result = loader.load_image(str(path))
        if not result.get("success"):
            return None
        # Repeat single image to fill 3×4 grid
        img = result["data"]
        frames = [{"image": img} for _ in range(12)]
    else:
        # Assume video
        result = loader.load_video(str(path), max_frames=max_frames, sample_evenly=True)
        if not result.get("success") or not result.get("frames"):
            return None
        frames = result["frames"]

    grid = grid_creator.create_grid(frames, location_text=location_text or path.name)
    if grid is None:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    grid_path = Path(tmp.name)
    grid_creator.save_grid(grid, str(grid_path))
    return grid_path


def analyze(
    file_path: str,
    context: Optional[Dict[str, Any]] = None,
    model_func: Optional[Callable[[str], Dict[str, Any]]] = None,
    location_text: str = "",
) -> Dict[str, Any]:
    """
    Load media, build 3×4 grid, call analyzer, return normalized incident dict.

    Args:
        file_path: Path to image or video
        context: Optional {time_of_day, weather, location}
        model_func: Optional; if None, uses model stub (or real VLM when wired)
        location_text: Optional overlay for grid

    Returns:
        Normalized incident dict suitable for DecisionPipeline.process
    """
    grid_path = load_and_build_grid(file_path, location_text=location_text)
    if grid_path is None:
        return {
            "incident_type": "unknown",
            "confidence": 0.0,
            "vehicles": {"total_count": 0, "involved_in_incident": [], "nearby_traffic": [], "types": []},
            "lanes": {"total": 0, "blocked": [], "shoulder_blocked": False},
            "hazards": {"fire": False, "smoke": False, "debris": False, "injuries_visible": False, "spill": False},
            "traffic": {"state": "unknown", "queue_length_meters": 0},
            "emergency_response": {
                "police_present": False, "officers_count": 0,
                "ambulance_present": False, "fire_truck_present": False, "responders_on_scene": False,
            },
            "analysis_failed": True,
        }

    model = model_func or _get_default_model()
    result = model(str(grid_path))

    try:
        grid_path.unlink(missing_ok=True)
    except Exception:
        pass

    return result
