"""
Demo script for adversarial scenario generation.

This script demonstrates:
1. Normal scenario generation
2. Adversarial scenario generation
3. Mixed mode scenarios
4. Difficulty progression
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.environments.adversarial_scenarios import (
    AdversarialScenarioGenerator,
    AdversarialEventType,
)


def demo_scenario_generation():
    """Demonstrate scenario generation capabilities."""
    
    print("=" * 60)
    print("Adversarial Scenario Generator Demo")
    print("=" * 60)
    
    # 1. Normal Mode
    print("\n1. NORMAL MODE (Baseline Training)")
    print("-" * 60)
    
    generator = AdversarialScenarioGenerator(mode="normal", difficulty="medium", seed=42)
    
    for i in range(3):
        scenario = generator.sample_scenario()
        print(f"\nScenario {i+1}:")
        print(f"  Adversarial: {scenario.is_adversarial}")
        print(f"  Vehicles: {scenario.num_vehicles}")
        print(f"  Traffic Density: {scenario.traffic_density}")
        print(f"  Adversarial Events: {len(scenario.adversarial_events)}")
    
    # 2. Adversarial Mode
    print("\n\n2. ADVERSARIAL MODE")
    print("-" * 60)
    
    generator = AdversarialScenarioGenerator(mode="adversarial", difficulty="medium", seed=42)
    
    for i in range(3):
        scenario = generator.sample_scenario()
        print(f"\nScenario {i+1}:")
        print(f"  Adversarial: {scenario.is_adversarial}")
        print(f"  Vehicles: {scenario.num_vehicles}")
        print(f"  Traffic Density: {scenario.traffic_density}")
        print(f"  Adversarial Events: {len(scenario.adversarial_events)}")
        
        if scenario.adversarial_events:
            print("  Events:")
            for j, event in enumerate(scenario.adversarial_events):
                print(f"    {j+1}. {event.event_type.value}")
                print(f"       Trigger: {event.trigger_time:.1f}s")
                print(f"       Severity: {event.severity:.2f}")
                print(f"       Duration: {event.duration:.1f}s")
    
    # 3. Mixed Mode
    print("\n\n3. MIXED MODE (50% Adversarial)")
    print("-" * 60)
    
    generator = AdversarialScenarioGenerator(
        mode="mixed",
        difficulty="medium",
        adversarial_ratio=0.5,
        seed=42
    )
    
    adversarial_count = 0
    for i in range(10):
        scenario = generator.sample_scenario()
        if scenario.is_adversarial:
            adversarial_count += 1
    
    print(f"Generated 10 scenarios:")
    print(f"  Adversarial: {adversarial_count}")
    print(f"  Normal: {10 - adversarial_count}")
    print(f"  Ratio: {adversarial_count / 10:.1%}")
    
    # 4. Difficulty Progression
    print("\n\n4. DIFFICULTY PROGRESSION")
    print("-" * 60)
    
    difficulties = ["easy", "medium", "hard", "extreme"]
    
    for difficulty in difficulties:
        generator = AdversarialScenarioGenerator(
            mode="adversarial",
            difficulty=difficulty,
            seed=42
        )
        
        # Generate sample and compute average severity
        scenarios = [generator.sample_scenario() for _ in range(5)]
        
        avg_events = np.mean([len(s.adversarial_events) for s in scenarios])
        
        all_severities = []
        for s in scenarios:
            all_severities.extend([e.severity for e in s.adversarial_events])
        
        avg_severity = np.mean(all_severities) if all_severities else 0.0
        
        print(f"\n{difficulty.upper()}:")
        print(f"  Avg Events per Scenario: {avg_events:.1f}")
        print(f"  Avg Event Severity: {avg_severity:.2f}")
    
    # 5. Event Type Distribution
    print("\n\n5. EVENT TYPE DISTRIBUTION")
    print("-" * 60)
    
    generator = AdversarialScenarioGenerator(
        mode="adversarial",
        difficulty="medium",
        seed=42
    )
    
    event_counts = {event_type: 0 for event_type in AdversarialEventType}
    
    for _ in range(50):
        scenario = generator.sample_scenario()
        for event in scenario.adversarial_events:
            event_counts[event.event_type] += 1
    
    print("\nEvent Type Frequency (50 scenarios):")
    for event_type, count in event_counts.items():
        print(f"  {event_type.value}: {count}")
    
    # 6. Statistics
    print("\n\n6. GENERATOR STATISTICS")
    print("-" * 60)
    
    stats = generator.get_stats()
    print(f"\nTotal Scenarios Generated: {stats['total_scenarios']}")
    print(f"Adversarial Scenarios: {stats['adversarial_scenarios']}")
    print(f"Adversarial Percentage: {stats['adversarial_percentage']:.1f}%")
    print(f"Mode: {stats['mode']}")
    print(f"Difficulty: {stats['difficulty']}")


if __name__ == "__main__":
    demo_scenario_generation()