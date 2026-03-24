# tests/test_decision_engine.py
from decision_engine.severity import calculate_severity
from compare_models import gpt4_analyze  # Use your stubs!

def test_severity_with_stub_output():
    # Get stub output
    output = gpt4_analyze("/path/to/Image_79.jpg")
    
    # Test decision engine
    severity = calculate_severity(output)
    assert severity['level'] in ['RED', 'YELLOW', 'GREEN']
    print(f"Severity: {severity}")