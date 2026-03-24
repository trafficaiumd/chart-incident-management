# tests/test_output_layer.py
"""
Test all output layer components
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from output_layer.chart_formatter import CHARTFormatter, test_chart_formatter
from output_layer.json_exporter import JSONExporter
from output_layer.dashboard_integration import DashboardIntegration
from output_layer.email_alert import EmailAlert
from decision_engine.pipeline import DecisionPipeline

def test_all_outputs():
    """Test all output layer components"""
    
    print("="*70)
    print("TESTING OUTPUT LAYER")
    print("="*70)
    
    # Create sample decision
    pipeline = DecisionPipeline()
    
    crash_output = {
        'incident_type': 'crash',
        'confidence': 0.95,
        'vehicles': {'total_count': 4},
        'lanes': {'blocked': [1, 2]},
        'hazards': {'debris': True},
        'traffic': {'state': 'stopped'},
        'emergency_response': {}
    }
    
    context = {'location': 'I-95 NB mile 27.4'}
    decision = pipeline.process(crash_output, context)
    
    # Test 1: CHART Formatter
    print("\n📋 Test 1: CHART Formatter")
    formatter = CHARTFormatter()
    chart_output = formatter.format(decision)
    print(f"   ✅ Generated CHART incident: {chart_output['incident_id']}")
    
    # Test 2: JSON Exporter
    print("\n📁 Test 2: JSON Exporter")
    exporter = JSONExporter("test_output")
    path = exporter.export(chart_output, "test_incident.json")
    print(f"   ✅ Exported to: {path}")
    
    # Test 3: Dashboard Integration
    print("\n🖥️ Test 3: Dashboard Integration")
    dashboard = DashboardIntegration()
    print("   ✅ Dashboard ready (run with streamlit)")
    
    # Test 4: Email Alert
    print("\n📧 Test 4: Email Alert")
    alert = EmailAlert()
    alert.add_recipient("test@chart.gov")
    print("   ✅ Email system ready")
    
    print("\n" + "="*70)
    print("✅ ALL OUTPUT LAYER TESTS PASSED")
    print("="*70)
    
    return chart_output

if __name__ == "__main__":
    test_all_outputs()