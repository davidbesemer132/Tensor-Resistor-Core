"""
TRC-132 Sovereign Kernel - TRC Logic Engine
============================================

This module implements the core logic engine for the Tensor Resistor Core (TRC-132)
system. It loads and enforces physics constraints defined in system_manifest.json
and provides the foundation for tensor-based resistor operations.

Author: TRC Development Team
Version: 1.0.0
Date: 2025-12-30
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import hashlib
from datetime import datetime


# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Enumerations and Constants
# ============================================================================

class OperationMode(Enum):
    """Enumeration of TRC operational modes."""
    SOVEREIGN = "sovereign"
    RESTRICTED = "restricted"
    DIAGNOSTIC = "diagnostic"
    FAILSAFE = "failsafe"


class ConstraintType(Enum):
    """Types of physics constraints enforced by the kernel."""
    IMPEDANCE = "impedance"
    FREQUENCY = "frequency"
    POWER = "power"
    VOLTAGE = "voltage"
    CURRENT = "current"
    THERMAL = "thermal"
    TEMPORAL = "temporal"
    DIMENSIONAL = "dimensional"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PhysicsConstraint:
    """Represents a single physics constraint from the manifest."""
    name: str
    constraint_type: ConstraintType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: str = ""
    description: str = ""
    critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, value: float) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against this constraint.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.min_value is not None and value < self.min_value:
            return False, f"Value {value} below minimum {self.min_value} {self.unit}"
        
        if self.max_value is not None and value > self.max_value:
            return False, f"Value {value} exceeds maximum {self.max_value} {self.unit}"
        
        return True, None


@dataclass
class SystemState:
    """Represents the current operational state of the TRC system."""
    mode: OperationMode = OperationMode.SOVEREIGN
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    active_constraints: Dict[str, bool] = field(default_factory=dict)
    last_validation: Optional[str] = None
    error_count: int = 0
    warning_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TRC Sovereign Kernel
# ============================================================================

class TRCSovereignKernel:
    """
    The core logic engine for TRC-132 system.
    
    This kernel:
    - Loads physics constraints from system_manifest.json
    - Enforces operational limits and safety constraints
    - Manages system state and operational modes
    - Provides constraint validation and reporting
    """

    def __init__(self, manifest_path: Optional[str] = None):
        """
        Initialize the TRC Sovereign Kernel.
        
        Args:
            manifest_path: Path to system_manifest.json file.
                         If None, attempts to locate it automatically.
        """
        self.manifest_path = manifest_path or self._find_manifest()
        self.constraints: Dict[str, PhysicsConstraint] = {}
        self.system_state = SystemState()
        self.manifest_data: Dict[str, Any] = {}
        self.kernel_id = self._generate_kernel_id()
        
        logger.info(f"TRC Sovereign Kernel initializing (ID: {self.kernel_id})")
        
        if self.manifest_path and os.path.exists(self.manifest_path):
            self.load_manifest(self.manifest_path)
        else:
            logger.warning("No system_manifest.json found. Kernel operating in minimal mode.")

    def _find_manifest(self) -> Optional[str]:
        """Attempt to locate system_manifest.json in standard locations."""
        search_paths = [
            "system_manifest.json",
            "config/system_manifest.json",
            "software/config/system_manifest.json",
            "../config/system_manifest.json",
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                logger.info(f"Found manifest at: {path}")
                return path
        
        return None

    def _generate_kernel_id(self) -> str:
        """Generate a unique identifier for this kernel instance."""
        timestamp = datetime.utcnow().isoformat()
        data = f"TRC-132-kernel-{timestamp}".encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def load_manifest(self, manifest_path: str) -> bool:
        """
        Load and parse the system_manifest.json file.
        
        Args:
            manifest_path: Path to the manifest file
            
        Returns:
            True if manifest loaded successfully, False otherwise
        """
        try:
            with open(manifest_path, 'r') as f:
                self.manifest_data = json.load(f)
            
            logger.info(f"Loaded manifest from {manifest_path}")
            
            # Parse constraints from manifest
            if 'constraints' in self.manifest_data:
                self._parse_constraints(self.manifest_data['constraints'])
            
            # Load system configuration
            if 'system_config' in self.manifest_data:
                self._load_system_config(self.manifest_data['system_config'])
            
            # Update last validation timestamp
            self.system_state.last_validation = datetime.utcnow().isoformat()
            
            logger.info(f"Manifest loaded successfully. {len(self.constraints)} constraints enforced.")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse manifest JSON: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading manifest: {e}")
            return False

    def _parse_constraints(self, constraints_data: Dict[str, Any]) -> None:
        """Parse constraint definitions from manifest data."""
        for constraint_name, constraint_def in constraints_data.items():
            try:
                # Determine constraint type
                constraint_type_str = constraint_def.get('type', 'IMPEDANCE').upper()
                constraint_type = ConstraintType[constraint_type_str]
                
                # Create constraint object
                constraint = PhysicsConstraint(
                    name=constraint_name,
                    constraint_type=constraint_type,
                    min_value=constraint_def.get('min'),
                    max_value=constraint_def.get('max'),
                    unit=constraint_def.get('unit', ''),
                    description=constraint_def.get('description', ''),
                    critical=constraint_def.get('critical', False),
                    metadata=constraint_def.get('metadata', {})
                )
                
                self.constraints[constraint_name] = constraint
                self.system_state.active_constraints[constraint_name] = True
                
                logger.debug(f"Loaded constraint: {constraint_name}")
                
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse constraint '{constraint_name}': {e}")

    def _load_system_config(self, config_data: Dict[str, Any]) -> None:
        """Load system configuration from manifest."""
        if 'operation_mode' in config_data:
            mode_str = config_data['operation_mode'].upper()
            try:
                self.system_state.mode = OperationMode[mode_str]
            except KeyError:
                logger.warning(f"Unknown operation mode: {mode_str}")
        
        if 'metadata' in config_data:
            self.system_state.metadata.update(config_data['metadata'])

    def validate_parameter(self, 
                          constraint_name: str, 
                          value: float) -> Tuple[bool, Optional[str]]:
        """
        Validate a parameter against a specific constraint.
        
        Args:
            constraint_name: Name of the constraint to validate against
            value: The value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if constraint_name not in self.constraints:
            error = f"Unknown constraint: {constraint_name}"
            logger.warning(error)
            return False, error
        
        constraint = self.constraints[constraint_name]
        is_valid, error_msg = constraint.validate(value)
        
        if not is_valid:
            self.system_state.error_count += 1
            logger.warning(f"Constraint violation - {constraint_name}: {error_msg}")
        
        return is_valid, error_msg

    def validate_parameters(self, parameters: Dict[str, float]) -> Tuple[bool, Dict[str, str]]:
        """
        Validate multiple parameters at once.
        
        Args:
            parameters: Dictionary of constraint_name -> value pairs
            
        Returns:
            Tuple of (all_valid, errors_dict)
        """
        errors = {}
        all_valid = True
        
        for constraint_name, value in parameters.items():
            is_valid, error_msg = self.validate_parameter(constraint_name, value)
            if not is_valid:
                all_valid = False
                errors[constraint_name] = error_msg
        
        return all_valid, errors

    def get_constraint(self, constraint_name: str) -> Optional[PhysicsConstraint]:
        """Retrieve a constraint by name."""
        return self.constraints.get(constraint_name)

    def get_all_constraints(self) -> Dict[str, PhysicsConstraint]:
        """Get all active constraints."""
        return self.constraints.copy()

    def get_critical_constraints(self) -> Dict[str, PhysicsConstraint]:
        """Get only critical constraints."""
        return {
            name: constraint 
            for name, constraint in self.constraints.items() 
            if constraint.critical
        }

    def set_operation_mode(self, mode: OperationMode) -> None:
        """Set the system operation mode."""
        old_mode = self.system_state.mode
        self.system_state.mode = mode
        logger.info(f"Operation mode changed: {old_mode.value} -> {mode.value}")

    def get_system_state(self) -> SystemState:
        """Get the current system state."""
        return self.system_state

    def get_status_report(self) -> Dict[str, Any]:
        """Generate a comprehensive status report."""
        return {
            'kernel_id': self.kernel_id,
            'timestamp': datetime.utcnow().isoformat(),
            'operation_mode': self.system_state.mode.value,
            'total_constraints': len(self.constraints),
            'active_constraints': sum(1 for v in self.system_state.active_constraints.values() if v),
            'critical_constraints': len(self.get_critical_constraints()),
            'error_count': self.system_state.error_count,
            'warning_count': self.system_state.warning_count,
            'last_validation': self.system_state.last_validation,
            'manifest_loaded': bool(self.manifest_data),
        }

    def reset_error_counters(self) -> None:
        """Reset error and warning counters."""
        self.system_state.error_count = 0
        self.system_state.warning_count = 0
        logger.info("Error counters reset")

    def export_constraints_json(self) -> str:
        """Export all constraints as JSON."""
        constraints_dict = {}
        for name, constraint in self.constraints.items():
            constraints_dict[name] = {
                'type': constraint.constraint_type.value,
                'min': constraint.min_value,
                'max': constraint.max_value,
                'unit': constraint.unit,
                'description': constraint.description,
                'critical': constraint.critical,
                'metadata': constraint.metadata,
            }
        return json.dumps(constraints_dict, indent=2)


# ============================================================================
# Module-Level Functions
# ============================================================================

def create_kernel(manifest_path: Optional[str] = None) -> TRCSovereignKernel:
    """
    Factory function to create and initialize a TRC Sovereign Kernel.
    
    Args:
        manifest_path: Optional path to system_manifest.json
        
    Returns:
        Initialized TRCSovereignKernel instance
    """
    return TRCSovereignKernel(manifest_path)


def validate_operation(kernel: TRCSovereignKernel,
                      parameters: Dict[str, float]) -> Tuple[bool, Dict[str, str]]:
    """
    Validate an operation against all active constraints.
    
    Args:
        kernel: TRCSovereignKernel instance
        parameters: Dictionary of parameter values to validate
        
    Returns:
        Tuple of (operation_allowed, error_details)
    """
    return kernel.validate_parameters(parameters)


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("TRC-132 Sovereign Kernel - Initialization Test")
    print("=" * 70)
    
    # Create kernel
    kernel = create_kernel()
    
    # Print status
    status = kernel.get_status_report()
    print("\nKernel Status Report:")
    print(json.dumps(status, indent=2))
    
    print("\nActive Constraints:")
    constraints = kernel.get_all_constraints()
    if constraints:
        for name, constraint in constraints.items():
            print(f"  - {name}: {constraint.constraint_type.value} "
                  f"({constraint.min_value} to {constraint.max_value} {constraint.unit})")
    else:
        print("  No constraints loaded. Manifest may be missing.")
    
    print("\nCritical Constraints:")
    critical = kernel.get_critical_constraints()
    if critical:
        for name in critical:
            print(f"  - {name}")
    else:
        print("  No critical constraints defined.")
    
    print("\n" + "=" * 70)
