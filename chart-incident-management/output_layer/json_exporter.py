# output_layer/json_exporter.py
"""
JSON Export functionality
Saves incident data to files for logging/analysis
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class JSONExporter:
    """
    Exports incident data to JSON files
    """
    
    def __init__(self, output_dir: str = "incident_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export data to JSON file
        
        Args:
            data: Dictionary to export
            filename: Optional filename (auto-generated if not provided)
        
        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            incident_id = data.get('incident_id', data.get('incident', {}).get('type', 'unknown'))
            filename = f"incident_{incident_id}_{timestamp}.json"
        
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def export_batch(self, data_list: list, prefix: str = "batch") -> list:
        """Export multiple incidents"""
        paths = []
        for i, data in enumerate(data_list):
            path = self.export(data, f"{prefix}_{i:03d}.json")
            paths.append(path)
        return paths
    
    def load(self, filepath: str) -> Dict[str, Any]:
        """Load incident from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)


def test_exporter():
    """Test JSON exporter"""
    from chart_formatter import test_chart_formatter
    
    # Get sample data
    chart_data = test_chart_formatter()
    
    # Export
    exporter = JSONExporter("test_output")
    path = exporter.export(chart_data, "test_incident.json")
    
    print(f"\n✅ Exported to: {path}")
    
    # Load and verify
    loaded = exporter.load(path)
    print(f"✅ Loaded incident: {loaded['incident_id']}")

if __name__ == "__main__":
    test_exporter()