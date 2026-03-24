# output_layer/dashboard_integration.py
"""
Integration with Streamlit demo dashboard
"""
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create dummy st object to prevent crashes
    class DummySt:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    st = DummySt()



from datetime import datetime
from typing import Dict, Any

class DashboardIntegration:
    """
    Connects decision engine to Streamlit dashboard
    """
    
    def __init__(self):
        self.incident_history = []
    
    def display_incident(self, decision: Dict[str, Any], chart_format: Dict[str, Any]):
        """Display incident in Streamlit dashboard"""
        
        # Add to history
        self.incident_history.append({
            'timestamp': datetime.now(),
            'decision': decision,
            'chart': chart_format
        })
        
        # Show current incident
        st.title("🚦 CHART Traffic Incident Management")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity = decision['severity']['level']
            color = "red" if severity == "RED" else "orange" if severity == "YELLOW" else "green"
            st.markdown(f"### Severity: :{color}[{severity}]")
            st.markdown(f"Score: {decision['severity']['score']}")
        
        with col2:
            st.markdown("### Resources")
            for resource, needed in decision['resources'].items():
                if resource != 'reasons' and needed:
                    st.markdown(f"- {resource.upper()}")
        
        with col3:
            st.markdown("### Secondary Risk")
            risk = decision['secondary_risk']['risk_level']
            risk_color = "red" if risk == "HIGH" else "orange" if risk == "MEDIUM" else "green"
            st.markdown(f":{risk_color}[{risk}] ({decision['secondary_risk']['risk_score']})")
        
        st.markdown("---")
        
        # VMS Display
        st.subheader("📋 VMS Messages")
        vms = decision['vms']
        st.code(f"""
╔══════════════════════════════╗
║  {vms['primary']:<26} ║
║  {vms['secondary']:<26} ║
║  {vms['tertiary']:<26} ║
╚══════════════════════════════╝
        """)
        
        # CHART Output
        with st.expander("📄 CHART Formatted Output"):
            st.json(chart_format)
        
        # Review status
        if decision['requires_human_review']:
            st.warning("👤 Requires Human Review")
        else:
            st.success("✅ Auto-approved")
        
        # History
        with st.expander("📜 Incident History"):
            for i, incident in enumerate(self.incident_history[-5:]):  # Last 5
                st.text(f"{incident['timestamp'].strftime('%H:%M:%S')}: {incident['decision']['incident']['type']}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        if not self.incident_history:
            return {}
        
        severities = [i['decision']['severity']['level'] for i in self.incident_history]
        risks = [i['decision']['secondary_risk']['risk_level'] for i in self.incident_history]
        
        return {
            'total_incidents': len(self.incident_history),
            'severity_counts': {
                'RED': severities.count('RED'),
                'YELLOW': severities.count('YELLOW'),
                'GREEN': severities.count('GREEN')
            },
            'risk_counts': {
                'HIGH': risks.count('HIGH'),
                'MEDIUM': risks.count('MEDIUM'),
                'LOW': risks.count('LOW')
            }
        }


def demo_dashboard():
    """Run a demo of the dashboard integration"""
    from decision_engine.pipeline import DecisionPipeline
    from chart_formatter import CHARTFormatter
    
    # Initialize
    pipeline = DecisionPipeline()
    formatter = CHARTFormatter()
    dashboard = DashboardIntegration()
    
    # Sample incident
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
    chart_format = formatter.format(decision)
    
    # Display
    dashboard.display_incident(decision, chart_format)
    
    # Show stats
    st.sidebar.markdown("## 📊 Stats")
    stats = dashboard.get_stats()
    st.sidebar.json(stats)

if __name__ == "__main__":
    # This would be run with: streamlit run output_layer/dashboard_integration.py
    demo_dashboard()