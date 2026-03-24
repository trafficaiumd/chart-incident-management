#tests/evaluate.py
"""
Model evaluation module for testing vision-language models on incident detection.
"""

import time
from typing import Dict, Any, List, Callable, Tuple, Optional
from test_dataset import TestCase


class ModelEvaluator:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []

    def evaluate_model(
        self,
        model_func: Callable[[str], Dict[str, Any]],
        test_cases: List[TestCase],
        fields_to_score: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run model on all test cases and calculate metrics.

        Args:
            model_func: Function that takes image path and returns normalized schema dict
            test_cases: List of test cases to evaluate
            fields_to_score: List of dotted paths in the normalized schema, e.g.:
                - "incident_type"
                - "vehicles.total_count"
                - "lanes.blocked"
                - "hazards.fire"
        
        Returns:
            Dictionary containing evaluation metrics and detailed results
        """
        if fields_to_score is None:
            fields_to_score = [
                "incident_type",
                "vehicles.total_count",
                "lanes.blocked",
                "hazards.fire",
                "hazards.smoke",
            ]

        self.results = []
        total_cost = 0.0
        total_time = 0.0

        # Initialize per-field statistics
        field_stats: Dict[str, Dict[str, float]] = {
            f: {"correct": 0, "total": 0, "unknown": 0} for f in fields_to_score
        }

        # Initialize error breakdown counters
        error_breakdown = {
            "incident_type": 0,
            "vehicle_count": 0,
            "vehicle_types": 0,
            "lane_blocked": 0,
            "hazards": 0,
            "traffic_state": 0,
            "emergency_response": 0,
            "pedestrians": 0,
        }

        print(f"\nEvaluating {len(test_cases)} test cases...")

        for i, case in enumerate(test_cases):
            print(f"  [{i+1}/{len(test_cases)}] {case.description}")
            start = time.time()

            try:
                output = model_func(case.image_path)
                elapsed = time.time() - start
            except Exception as e:
                print(f"    ❌ Error: {e}")
                output = {"incident_type": "error", "error": str(e)}
                elapsed = 0.0

            cost = 0.02  # placeholder cost per inference

            # Compare outputs and collect errors
            is_correct, errors = self.compare_outputs(output, case.expected)

            # Update error breakdown
            for key in errors:
                if key in error_breakdown:
                    error_breakdown[key] += 1

            # Update per-field statistics
            self._update_field_stats(field_stats, fields_to_score, output, case.expected)

            print("    ✅ Correct" if is_correct else f"    ❌ Incorrect: {errors}")

            # Store detailed result
            self.results.append(
                {
                    "case": case.description,
                    "image": case.image_path,
                    "correct": is_correct,
                    "errors": errors,
                    "output": output,
                    "expected": case.expected,
                    "time": elapsed,
                    "cost": cost,
                }
            )

            total_time += elapsed
            total_cost += cost

        # Calculate overall metrics
        overall_accuracy = (
            sum(1 for r in self.results if r["correct"]) / len(self.results)
            if self.results
            else 0.0
        )

        per_field = self._summarize_field_stats(field_stats)

        return {
            "accuracy": overall_accuracy,
            "per_field": per_field,
            "error_breakdown": error_breakdown,
            "avg_time": total_time / len(test_cases) if test_cases else 0.0,
            "total_cost": total_cost,
            "total_tests": len(test_cases),
            "correct": sum(1 for r in self.results if r["correct"]),
            "detailed_results": self.results,
        }

    # ----- helpers for per-field stats ----------------------------------

    def _get_dotted(self, obj: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dotted path notation."""
        cur: Any = obj
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    def _update_field_stats(
        self,
        stats: Dict[str, Dict[str, float]],
        fields: List[str],
        output: Dict[str, Any],
        expected: Dict[str, Any],
    ) -> None:
        """Update per-field statistics based on comparison results."""
        for field in fields:
            s = stats[field]
            exp_val = self._get_dotted(expected, field)
            out_val = self._get_dotted(output, field)

            s["total"] += 1

            if isinstance(out_val, str) and out_val.upper() == "UNKNOWN":
                s["unknown"] += 1

            if out_val == exp_val:
                s["correct"] += 1

    def _summarize_field_stats(
        self,
        stats: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Convert raw field statistics to summary metrics."""
        summary: Dict[str, Dict[str, float]] = {}
        for field, s in stats.items():
            total = s["total"] or 1
            summary[field] = {
                "accuracy": s["correct"] / total,
                "unknown_rate": s["unknown"] / total,
                "total": total,
            }
        return summary

    # ----- existing structured comparison -------------------------------

    def compare_outputs(
        self, output: Dict[str, Any], expected: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Compare model output against expected values.
        
        Returns:
            Tuple of (is_correct, errors_dict)
        """
        errors: Dict[str, Any] = {}

        # Compare incident type
        if output.get("incident_type") != expected.get("incident_type"):
            errors["incident_type"] = {
                "expected": expected.get("incident_type"),
                "got": output.get("incident_type"),
            }

        # Compare vehicle information
        output_vehicles = output.get("vehicles", {}) or {}
        expected_vehicles = expected.get("vehicles", {}) or {}

        if output_vehicles.get("total_count") != expected_vehicles.get("total_count"):
            errors["vehicle_count"] = {
                "expected": expected_vehicles.get("total_count"),
                "got": output_vehicles.get("total_count"),
            }

        output_involved = len(output_vehicles.get("involved_in_incident", []) or [])
        expected_involved = len(expected_vehicles.get("involved_in_incident", []) or [])
        if output_involved != expected_involved:
            errors["involved_vehicles"] = {
                "expected": expected_involved,
                "got": output_involved,
            }

        output_nearby = len(output_vehicles.get("nearby_traffic", []) or [])
        expected_nearby = len(expected_vehicles.get("nearby_traffic", []) or [])
        if output_nearby != expected_nearby:
            errors["nearby_traffic"] = {
                "expected": expected_nearby,
                "got": output_nearby,
            }

        # Compare lane information
        output_lanes = output.get("lanes", {}) or {}
        expected_lanes = expected.get("lanes", {}) or {}
        if set(output_lanes.get("blocked", []) or []) != set(
            expected_lanes.get("blocked", []) or []
        ):
            errors["lane_blocked"] = {
                "expected": expected_lanes.get("blocked"),
                "got": output_lanes.get("blocked"),
            }

        if output_lanes.get("shoulder_blocked") != expected_lanes.get(
            "shoulder_blocked"
        ):
            errors["shoulder_blocked"] = {
                "expected": expected_lanes.get("shoulder_blocked"),
                "got": output_lanes.get("shoulder_blocked"),
            }

        # Compare hazard information
        output_hazards = output.get("hazards", {}) or {}
        expected_hazards = expected.get("hazards", {}) or {}
        hazard_errors: Dict[str, Any] = {}
        for key in ["fire", "smoke", "debris", "injuries_visible", "spill"]:
            if output_hazards.get(key) != expected_hazards.get(key):
                hazard_errors[key] = {
                    "expected": expected_hazards.get(key),
                    "got": output_hazards.get(key),
                }
        if hazard_errors:
            errors["hazards"] = hazard_errors

        # Compare traffic state
        output_traffic = output.get("traffic", {}) or {}
        expected_traffic = expected.get("traffic", {}) or {}
        if output_traffic.get("state") != expected_traffic.get("state"):
            errors["traffic_state"] = {
                "expected": expected_traffic.get("state"),
                "got": output_traffic.get("state"),
            }

        # Compare emergency response information
        output_emergency = output.get("emergency_response", {}) or {}
        expected_emergency = expected.get("emergency_response", {}) or {}
        emergency_errors: Dict[str, Any] = {}
        for key in [
            "police_present",
            "ambulance_present",
            "fire_truck_present",
            "responders_on_scene",
        ]:
            if output_emergency.get(key) != expected_emergency.get(key):
                emergency_errors[key] = {
                    "expected": expected_emergency.get(key),
                    "got": output_emergency.get(key),
                }

        if output_emergency.get("officers_count") != expected_emergency.get(
            "officers_count"
        ):
            emergency_errors["officers_count"] = {
                "expected": expected_emergency.get("officers_count"),
                "got": output_emergency.get("officers_count"),
            }

        if emergency_errors:
            errors["emergency_response"] = emergency_errors

        # Compare pedestrian information
        output_pedestrians = output.get("pedestrians", {}) or {}
        expected_pedestrians = expected.get("pedestrians", {}) or {}
        if output_pedestrians.get("count") != expected_pedestrians.get("count"):
            errors["pedestrians_count"] = {
                "expected": expected_pedestrians.get("count"),
                "got": output_pedestrians.get("count"),
            }

        return len(errors) == 0, errors

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a formatted evaluation report."""
        report = f"""
╔════════════════════════════════════════════╗
║         MODEL EVALUATION REPORT            ║
╚════════════════════════════════════════════╝

📊 OVERALL METRICS:
   Accuracy:          {results['accuracy']:.1%}
   Tests Passed:      {results['correct']}/{results['total_tests']}
   Average Time:      {results['avg_time']:.2f}s
   Total Cost:        ${results['total_cost']:.2f}

📊 PER-FIELD METRICS:
"""
        for field, stats in results.get("per_field", {}).items():
            report += f"   {field}: accuracy={stats['accuracy']:.1%}, unknown={stats['unknown_rate']:.1%}\n"

        report += "\n❌ ERROR BREAKDOWN:\n"
        for error_type, count in results["error_breakdown"].items():
            if count > 0:
                report += f"   {error_type}: {count} errors\n"

        report += "\n📋 DETAILED RESULTS:\n"
        for r in results["detailed_results"]:
            status = "✅" if r["correct"] else "❌"
            report += f"   {status} {r['case']}\n"
            if not r["correct"]:
                for error, details in r["errors"].items():
                    report += f"       - {error}: {details}\n"

        return report