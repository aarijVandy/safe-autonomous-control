"""
Configuration file for NADE highway environment scenarios.
"""

# Normal traffic configuration
NORMAL_CONFIG = {
    "name": "normal",
    "adversarial_mode": False,
    "vehicles_count": 15,
    "duration": 100.0,  
    "description": "Normal highway traffic with cooperative IDM vehicles",
}

# Mild adversarial configuration
MILD_ADVERSARIAL_CONFIG = {
    "name": "mild_adversarial",
    "adversarial_mode": True,
    "adversarial_intensity": 0.3,
    "vehicles_count": 20,
    "duration": 100.0,  
    "description": "Mildly adversarial traffic with occasional aggressive behaviors",
}

# Moderate adversarial configuration
MODERATE_ADVERSARIAL_CONFIG = {
    "name": "moderate_adversarial",
    "adversarial_mode": True,
    "adversarial_intensity": 0.5,
    "vehicles_count": 20,
    "duration": 100.0,
    "description": "Moderately adversarial traffic with frequent aggressive behaviors",
}

# Severe adversarial configuration
SEVERE_ADVERSARIAL_CONFIG = {
    "name": "severe_adversarial",
    "adversarial_mode": True,
    "adversarial_intensity": 0.8,
    "vehicles_count": 25,
    "duration": 100.0,  
    "description": "Severely adversarial traffic with very frequent aggressive behaviors",
}

# All configurations
SCENARIO_CONFIGS = {
    "normal": NORMAL_CONFIG,
    "mild": MILD_ADVERSARIAL_CONFIG,
    "moderate": MODERATE_ADVERSARIAL_CONFIG,
    "severe": SEVERE_ADVERSARIAL_CONFIG,
}


def get_scenario_config(scenario_name: str = "normal"):
    """
    Get configuration for a specific scenario.
    
    Args:
        scenario_name: Name of the scenario ('normal', 'mild', 'moderate', 'severe')
    
    Returns:
        config: Configuration dictionary for the scenario
    
    Raises:
        ValueError: If scenario_name is not recognized
    """
    if scenario_name not in SCENARIO_CONFIGS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available scenarios: {list(SCENARIO_CONFIGS.keys())}"
        )
    
    return SCENARIO_CONFIGS[scenario_name].copy()
