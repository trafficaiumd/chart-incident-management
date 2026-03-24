# ai_layer/json_parser.py

import re
import json
import logging
from typing import Any, Dict


UNKNOWN = "UNKNOWN"


class JSONParser:
    """
    Parser / normalizer for VLM JSON responses.

    - Strips markdown/code fences
    - Fixes common JSON mistakes
    - Parses to Python dict
    - Normalizes into the internal CHART schema used by the decision engine
    - Enforces UNKNOWN / analysis_failed on hard failures
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def parse(self, raw_response: str) -> Dict[str, Any]:
        """
        Takes raw text from API, returns:
        {
          "success": bool,
          "data": <normalized_incident_dict or None>,
          "error": <str or None>
        }
        """
        if not raw_response:
            return self._unknown_incident("Empty response")

        cleaned = self._strip_markdown(raw_response)
        cleaned = self._fix_json_errors(cleaned)

        try:
            raw_data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse failed: {e}")
            return self._unknown_incident(f"Invalid JSON: {str(e)}")

        try:
            normalized = self._normalize_schema(raw_data)
        except Exception as e:
            self.logger.error(f"Schema normalization failed: {e}")
            return self._unknown_incident(f"Schema error: {str(e)}")

        return {
            "success": True,
            "data": normalized,
            "error": None,
        }

    # ------------------------------------------------------------------ #
    # Markdown / JSON cleanup
    # ------------------------------------------------------------------ #

    
    def _strip_markdown(self, text:str) -> str:
        """Remove ```json and ``` markers"""
        # Remove ```json
        text = re.sub(r'```json\s*', '', text)
        # Remove trailing ```
        text = re.sub(r'\s*```', '', text)
        # Remove any other markdown code blocks
        text = re.sub(r'```\s*\w*', '', text)
        return text.strip()
    
    def _fix_json_errors(self, text:str) -> str:
        """Fix common AI JSON mistakes and quirks."""
        # Replace single quotes with double quotes
        text = re.sub(r"'([^']*)'", r'"\1"', text)
        
        # Remove trailing commas in objects
        text = re.sub(r',\s*}', '}', text)
        
        # Remove trailing commas in arrays
        text = re.sub(r',\s*]', ']', text)
        
        # Add missing quotes around keys (if needed)
        text = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        
        return text
    
     # Schema normalization -----------------------------------------------

    def _normalize_schema(self, data: Any) -> Dict[str, Any]:
        """
        Normalize arbitrary model JSON into the internal incident schema.

        This is the ONLY place that knows about the VLM-side JSON format.
        Everything downstream (tests, decision engine, dashboard) should rely
        on the normalized schema only.
        """
        if not isinstance(data, dict):
            raise ValueError("Top‑level JSON is not an object")

        # Top‑level fields (UNKNOWN‑safe defaults)
        incident_type = str(data.get("incident_type", "unknown")).lower()
        if incident_type not in {"crash", "debris", "breakdown", "fire", "none", "unknown"}:
            incident_type = "unknown"

        confidence = data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        vehicles_raw = data.get("vehicles", {}) or {}
        lanes_raw = data.get("lanes", {}) or {}
        hazards_raw = data.get("hazards", {}) or {}
        traffic_raw = data.get("traffic", {}) or {}
        emergency_raw = data.get("emergency_response", {}) or {}

        vehicles = self._normalize_vehicles(vehicles_raw)
        lanes = self._normalize_lanes(lanes_raw)
        hazards = self._normalize_hazards(hazards_raw)
        traffic = self._normalize_traffic(traffic_raw)
        emergency = self._normalize_emergency(emergency_raw)

        return {
            "incident_type": incident_type,
            "confidence": confidence,
            "vehicles": vehicles,
            "lanes": lanes,
            "hazards": hazards,
            "traffic": traffic,
            "emergency_response": emergency,
            # If we got here, analysis succeeded (even if many UNKNOWNs).
            "analysis_failed": False,
        }

    def validate(self, data:Dict[str, Any]) -> Dict[str, Any]:
        """Normalize vehicles block.

        Supports both:
        - prompt_builder-style:
          { "count": int, "types": ["car", ...], "damaged_vehicles": [...] }
        - richer internal style:
          { "total_count": int, "involved_in_incident": [...], "nearby_traffic": [...] }
        """
        if not isinstance(v, dict):
            v = {}
        
        # Prefer explicit total_count, else fall back to count, else 0
        total_count = v.get("total_count", v.get("count", 0))
        try:
            total_count = int(total_count)
        except (TypeError, ValueError):
            total_count = 0

        # Rich lists if present; otherwise, construct minimal entries from types
        involved = v.get("involved_in_incident") or []
        nearby = v.get("nearby_traffic") or []

        if not involved and not nearby:
            # Coarse types list, e.g. ["car","truck"]
            types = v.get("types") or []
            if isinstance(types, list):
                # Represent all as nearby_traffic; CHART will still see counts and types
                nearby = [{"type": str(t), "damaged": False} for t in types]

        # Basic shape guarantees
        if not isinstance(involved, list):
            involved = []
        if not isinstance(nearby, list):
            nearby = []

        return {
            "total_count": total_count,
            "involved_in_incident": involved,
            "nearby_traffic": nearby,
            # keep original coarse fields if present (useful for debugging/eval)
            "types": v.get("types", []),
        }

    def _normalize_lanes(self, l: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(l, dict):
            l = {}
        total = l.get("total", 0)
        try:
            total = int(total)
        except (TypeError, ValueError):
            total = 0

        blocked = l.get("blocked") or []
        if not isinstance(blocked, list):
            blocked = []

        blocked_clean = []
        for b in blocked:
            try:
                blocked_clean.append(int(b))
            except (TypeError, ValueError):
                continue

        shoulder_blocked = bool(l.get("shoulder_blocked", False))

        return {
            "total": total,
            "blocked": blocked_clean,
            "shoulder_blocked": shoulder_blocked,
        }

    def _normalize_hazards(self, h: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(h, dict):
            h = {}

        def b(key: str) -> bool:
            return bool(h.get(key, False))

        return {
            "fire": b("fire"),
            "smoke": b("smoke"),
            "debris": b("debris"),
            "injuries_visible": b("injuries_visible"),
            "spill": b("spill"),
        }

    def _normalize_traffic(self, t: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(t, dict):
            t = {}

        state = str(t.get("state", "unknown")).lower()
        if state not in {"flowing", "slowing", "stopped", "unknown"}:
            state = "unknown"

        queue = t.get("queue_length_meters", 0)
        try:
            queue = int(queue)
        except (TypeError, ValueError):
            queue = 0

        return {
            "state": state,
            "queue_length_meters": queue,
        }

    def _normalize_emergency(self, e: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(e, dict):
            e = {}

        def b(key: str) -> bool:
            return bool(e.get(key, False))

        officers = e.get("officers_count", 0)
        try:
            officers = int(officers)
        except (TypeError, ValueError):
            officers = 0

        return {
            "police_present": b("police_present"),
            "officers_count": officers,
            "ambulance_present": b("ambulance_present"),
            "fire_truck_present": b("fire_truck_present"),
            "responders_on_scene": b("responders_on_scene"),
        }

    # UNKNOWN / error helpers --------------------------------------------

    def _unknown_incident(self, message: str) -> Dict[str, Any]:
        """
        Fail‑safe: everything UNKNOWN / lowest‑risk, analysis_failed=True.
        """
        self.logger.warning(f"Falling back to UNKNOWN incident: {message}")

        data = {
            "incident_type": "unknown",
            "confidence": 0.0,
            "vehicles": {
                "total_count": 0,
                "involved_in_incident": [],
                "nearby_traffic": [],
                "types": [],
            },
            "lanes": {
                "total": 0,
                "blocked": [],
                "shoulder_blocked": False,
            },
            "hazards": {
                "fire": False,
                "smoke": False,
                "debris": False,
                "injuries_visible": False,
                "spill": False,
            },
            "traffic": {
                "state": "unknown",
                "queue_length_meters": 0,
            },
            "emergency_response": {
                "police_present": False,
                "officers_count": 0,
                "ambulance_present": False,
                "fire_truck_present": False,
                "responders_on_scene": False,
            },
            "analysis_failed": True,
        }

        return {
            "success": False,
            "data": data,
            "error": message,
        }

    def error_response(self, message:str) -> Dict[str, Any]:
        # Backwards‑compatible alias if you still use error_response elsewhere
        return self._unknown_incident(message)
        