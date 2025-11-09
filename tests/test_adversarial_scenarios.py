"""
Tests for adversarial scenario generation.
"""

import pytest
import numpy as np
from src.environments.adversarial_scenarios import (
    AdversarialScenarioGenerator,
    AdversarialEventType,
    DifficultyLevel,
    ScenarioMetrics,
)


class TestAdversarialScenarioGenerator:
    """Test suite for AdversarialScenarioGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = AdversarialScenarioGenerator(mode="normal", difficulty="medium")
        
        assert generator.mode == "normal"
        assert generator.difficulty == DifficultyLevel.MEDIUM
        assert generator.scenarios_generated == 0
    
    def test_normal_mode_scenarios(self):
        """Test that normal mode generates only normal scenarios."""
        generator = AdversarialScenarioGenerator(mode="normal", difficulty="easy", seed=42)
        
        for _ in range(10):
            scenario = generator.sample_scenario()
            assert not scenario.is_adversarial
            assert len(scenario.adversarial_events) == 0
    
    def test_adversarial_mode_scenarios(self):
        """Test that adversarial mode generates adversarial scenarios."""
        generator = AdversarialScenarioGenerator(mode="adversarial", difficulty="medium", seed=42)
        
        for _ in range(10):
            scenario = generator.sample_scenario()
            assert scenario.is_adversarial
    
    def test_mixed_mode_ratio(self):
        """Test that mixed mode respects adversarial ratio."""
        generator = AdversarialScenarioGenerator(
            mode="mixed",
            difficulty="medium",
            adversarial_ratio=0.5,
            seed=42
        )
        
        num_scenarios = 100
        adversarial_count = 0
        
        for _ in range(num_scenarios):
            scenario = generator.sample_scenario()
            if scenario.is_adversarial:
                adversarial_count += 1
        
        # Should be approximately 50% (with some variance)
        assert 35 < adversarial_count < 65
    
    def test_difficulty_scaling(self):
        """Test that difficulty affects event severity."""
        easy_gen = AdversarialScenarioGenerator(mode="adversarial", difficulty="easy", seed=42)
        hard_gen = AdversarialScenarioGenerator(mode="adversarial", difficulty="hard", seed=42)
        
        easy_scenario = easy_gen.sample_scenario()
        hard_scenario = hard_gen.sample_scenario()
        
        # Hard scenarios should have more events or higher severity
        if len(easy_scenario.adversarial_events) > 0 and len(hard_scenario.adversarial_events) > 0:
            easy_severity = np.mean([e.severity for e in easy_scenario.adversarial_events])
            hard_severity = np.mean([e.severity for e in hard_scenario.adversarial_events])
            
            assert hard_severity >= easy_severity
    
    def test_vehicle_positions_validity(self):
        """Test that generated vehicle positions are valid."""
        generator = AdversarialScenarioGenerator(mode="normal", seed=42)
        
        for _ in range(10):
            scenario = generator.sample_scenario()
            
            # Check that we have positions for all vehicles
            assert len(scenario.vehicle_positions) == scenario.num_vehicles
            
            # Check that ego vehicle is at position 0
            assert scenario.vehicle_positions[0][0] == 0.0
            
            # Check that lane indices are valid (0-3 for 4-lane highway)
            for pos, lane in scenario.vehicle_positions:
                assert 0 <= lane < 4
    
    def test_event_timing(self):
        """Test that events are scheduled at valid times."""
        generator = AdversarialScenarioGenerator(
            mode="adversarial",
            difficulty="medium",
            episode_length=1000,
            seed=42
        )
        
        scenario = generator.sample_scenario()
        
        for event in scenario.adversarial_events:
            # Events should be between 20% and 80% of episode
            assert 200 <= event.trigger_time <= 800
            
            # Duration should be positive
            assert event.duration > 0
            
            # Severity should be in [0, 1]
            assert 0 <= event.severity <= 1
    
    def test_stats_tracking(self):
        """Test that statistics are tracked correctly."""
        generator = AdversarialScenarioGenerator(
            mode="mixed",
            adversarial_ratio=0.5,
            seed=42
        )
        
        # Generate some scenarios
        for _ in range(20):
            generator.sample_scenario()
        
        stats = generator.get_stats()
        
        assert stats["total_scenarios"] == 20
        assert 0 < stats["adversarial_scenarios"] < 20
        assert 0 < stats["adversarial_percentage"] < 100


class TestScenarioMetrics:
    """Test suite for ScenarioMetrics."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = ScenarioMetrics()
        
        assert metrics.collisions == 0
        assert metrics.ttc_violations == 0
        assert metrics.timesteps == 0
    
    def test_collision_tracking(self):
        """Test collision tracking."""
        metrics = ScenarioMetrics()
        
        metrics.update(collision=True)
        metrics.update(collision=False)
        metrics.update(collision=True)
        
        assert metrics.collisions == 2
    
    def test_ttc_violation_tracking(self):
        """Test TTC violation tracking."""
        metrics = ScenarioMetrics()
        
        metrics.update(ttc=3.0, ttc_threshold=2.0)  # No violation
        metrics.update(ttc=1.5, ttc_threshold=2.0)  # Violation
        metrics.update(ttc=1.0, ttc_threshold=2.0)  # Violation
        
        assert metrics.ttc_violations == 2
    
    def test_jerk_accumulation(self):
        """Test jerk accumulation."""
        metrics = ScenarioMetrics()
        
        metrics.update(jerk=1.0)
        metrics.update(jerk=2.0)
        metrics.update(jerk=1.5)
        
        summary = metrics.get_summary()
        assert summary["avg_jerk"] == pytest.approx(1.5)
    
    def test_summary_statistics(self):
        """Test summary statistics computation."""
        metrics = ScenarioMetrics()
        
        for i in range(10):
            metrics.update(
                collision=(i < 2),  # 2 collisions
                ttc=2.5 - i * 0.3,
                ttc_threshold=2.0,
                jerk=0.5,
            )
        
        summary = metrics.get_summary()
        
        assert summary["collision_rate"] == 2
        assert summary["timesteps"] == 10
        assert 0 < summary["ttc_violation_rate"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])