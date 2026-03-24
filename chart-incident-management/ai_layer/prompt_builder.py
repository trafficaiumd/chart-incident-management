# ai_layer/prompt_builder.py
"""
Inject CHART business rules into prompts.
"""

class PromptBuilder:
    """
    Builds prompts with CHART rules embedded.
    """
    
    BASE_PROMPT = """
You are CHART's traffic incident analyzer. Analyze this traffic camera image.

CRITICAL CHART RULES (MUST FOLLOW):
- If you see a SCHOOL BUS involved: This is a RED severity incident
- If you see FIRE or SMOKE: This is a RED severity incident  
- If you see a TANKER TRUCK: This is a RED severity incident (hazmat risk)
- If 3+ VEHICLES involved: This is a RED severity incident
- If 2+ LANES blocked: This is a RED severity incident
- If unsure about ANY detail: Write "UNKNOWN" - DO NOT GUESS

Extract EXACTLY this JSON structure:
{
  "incident_type": "crash|debris|breakdown|fire|none",
  "confidence": 0.0-1.0,
  "vehicles": {
    "count": integer,
    "types": ["car", "truck", "bus", "tanker", "school_bus", "motorcycle"],
    "damaged_vehicles": []  // list of vehicle positions if visible
  },
  "lanes": {
    "total": integer,
    "blocked": [],  // lane numbers from left (1 = leftmost)
    "shoulder_blocked": boolean
  },
  "hazards": {
    "fire": boolean,
    "smoke": boolean,
    "debris": boolean,
    "injuries_visible": boolean
  },
  "traffic": {
    "state": "flowing|slowing|stopped",
    "queue_length_meters": integer  // approximate
  }
}

Return ONLY the JSON. No explanations, no markdown.
"""
    
    @classmethod
    def build_prompt(cls, context=None):
        """
        Build prompt with optional context injection
        """
        if not context:
            return cls.BASE_PROMPT
        
        # Add context-specific rules
        context_rules = []
        
        if context.get('time_of_day') == 'night':
            context_rules.append("- Night conditions: Pay attention to headlights, emergency vehicle lights")
            
        if context.get('weather') == 'rain':
            context_rules.append("- Rain conditions: Reduced visibility, look for hydroplaning, spin-outs")
            
        if context.get('near_school'):
            context_rules.append("- SCHOOL ZONE: Extra attention for school buses and children")
        
        if context_rules:
            return cls.BASE_PROMPT + "\n\nCONTEXT-SPECIFIC RULES:\n" + "\n".join(context_rules)
        
        return cls.BASE_PROMPT