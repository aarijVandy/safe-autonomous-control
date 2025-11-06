"""
Base controller interface for autonomous vehicle controllers.

This module defines the abstract base class that all controllers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseController(ABC):
    """
    Abstract base class for vehicle controllers.
    
    All controllers (PID, neural network, etc.) should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the controller.
        
        Args:
            config: Configuration dictionary for the controller
        """
        self.config = config or {}
        self.reset()
    
    @abstractmethod
    def compute_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> np.ndarray:
        """
        Compute the control action based on the observation.
        
        Args:
            observation: Current observation from the environment
            info: Additional information (optional)
        
        Returns:
            action: Control action to apply
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Reset the controller state.
        
        Called at the beginning of each episode.
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current controller parameters.
        
        Returns:
            params: Dictionary of controller parameters
        """
        return self.config.copy()
    
    def set_params(self, params: Dict[str, Any]):
        """
        Set controller parameters.
        
        Args:
            params: Dictionary of parameters to set
        """
        self.config.update(params)
