"""
Equilibrium Resistance Core Implementation
==========================================

A complete implementation of the Equilibrium Resistance Core system featuring:
- ActiveMaterial class for material property management
- TensorRegulator for tensor field regulation
- ResistanceComputer for resistance calculations
- Simulation runner for system dynamics

Author: davidbesemer132
Date: 2025-12-30
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod
import json
from datetime import datetime


@dataclass
class MaterialProperties:
    """Stores material properties for resistance calculations."""
    conductivity: float
    permeability: float
    permittivity: float
    temperature: float = 300.0  # Kelvin
    strain: float = 0.0
    defect_density: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert properties to dictionary."""
        return {
            'conductivity': self.conductivity,
            'permeability': self.permeability,
            'permittivity': self.permittivity,
            'temperature': self.temperature,
            'strain': self.strain,
            'defect_density': self.defect_density
        }


class ActiveMaterial:
    """
    Represents an active material with dynamic properties.
    
    Manages material state and provides methods for property calculation
    based on environmental and operational conditions.
    """
    
    def __init__(self, material_id: str, initial_properties: MaterialProperties):
        """
        Initialize an active material.
        
        Args:
            material_id: Unique identifier for the material
            initial_properties: Initial material properties
        """
        self.material_id = material_id
        self.properties = initial_properties
        self.history = []
        self.state_timestamp = datetime.utcnow()
        self._record_state()
    
    def _record_state(self):
        """Record current state to history."""
        self.history.append({
            'timestamp': self.state_timestamp.isoformat(),
            'properties': self.properties.to_dict()
        })
    
    def update_temperature(self, new_temperature: float, rate: float = 1.0):
        """
        Update material temperature with thermal dynamics.
        
        Args:
            new_temperature: Target temperature in Kelvin
            rate: Temperature change rate (0-1, where 1 is instantaneous)
        """
        old_temp = self.properties.temperature
        self.properties.temperature = (
            old_temp * (1 - rate) + new_temperature * rate
        )
        self.state_timestamp = datetime.utcnow()
        self._record_state()
    
    def apply_strain(self, strain_value: float):
        """
        Apply mechanical strain to the material.
        
        Args:
            strain_value: Strain magnitude (-1 to 1)
        """
        self.properties.strain = np.clip(strain_value, -1.0, 1.0)
        self.state_timestamp = datetime.utcnow()
        self._record_state()
    
    def increase_defect_density(self, increase: float):
        """
        Increase defect density due to radiation or degradation.
        
        Args:
            increase: Amount to increase defect density
        """
        self.properties.defect_density = min(
            self.properties.defect_density + increase, 1.0
        )
        self.state_timestamp = datetime.utcnow()
        self._record_state()
    
    def get_effective_conductivity(self) -> float:
        """
        Calculate effective conductivity considering all factors.
        
        Returns:
            Effective conductivity value
        """
        base_conductivity = self.properties.conductivity
        
        # Temperature effect (simplified)
        temp_factor = np.exp(-0.01 * (self.properties.temperature - 300.0))
        
        # Strain effect
        strain_factor = 1.0 - 0.5 * abs(self.properties.strain)
        
        # Defect effect
        defect_factor = 1.0 - 0.8 * self.properties.defect_density
        
        effective = base_conductivity * temp_factor * strain_factor * defect_factor
        return max(effective, 0.0)
    
    def get_status(self) -> Dict:
        """Get current material status."""
        return {
            'material_id': self.material_id,
            'properties': self.properties.to_dict(),
            'effective_conductivity': self.get_effective_conductivity(),
            'history_length': len(self.history),
            'timestamp': self.state_timestamp.isoformat()
        }


class TensorRegulator:
    """
    Regulates tensor fields in the resistor core.
    
    Maintains equilibrium of electromagnetic and mechanical tensor fields
    through active feedback and compensation.
    """
    
    def __init__(self, num_dimensions: int = 3, regulation_strength: float = 0.1):
        """
        Initialize tensor regulator.
        
        Args:
            num_dimensions: Number of spatial dimensions
            regulation_strength: Strength of regulation (0-1)
        """
        self.num_dimensions = num_dimensions
        self.regulation_strength = np.clip(regulation_strength, 0.0, 1.0)
        self.tensor_field = np.zeros((num_dimensions, num_dimensions))
        self.target_field = np.zeros((num_dimensions, num_dimensions))
        self.regulation_history = []
        self.cycle_count = 0
    
    def set_target_field(self, target_field: np.ndarray):
        """
        Set the target tensor field.
        
        Args:
            target_field: Target tensor field (must be square matrix)
        """
        if target_field.shape != (self.num_dimensions, self.num_dimensions):
            raise ValueError(f"Target field must be {self.num_dimensions}x{self.num_dimensions}")
        self.target_field = target_field.copy()
    
    def regulate_step(self) -> float:
        """
        Perform one regulation step.
        
        Returns:
            Field deviation magnitude
        """
        # Calculate deviation from target
        deviation = self.target_field - self.tensor_field
        deviation_magnitude = np.linalg.norm(deviation)
        
        # Apply regulation with strength scaling
        adjustment = deviation * self.regulation_strength
        self.tensor_field += adjustment
        
        # Record history
        self.regulation_history.append({
            'cycle': self.cycle_count,
            'deviation': float(deviation_magnitude),
            'field_norm': float(np.linalg.norm(self.tensor_field))
        })
        
        self.cycle_count += 1
        return deviation_magnitude
    
    def get_tensor_field(self) -> np.ndarray:
        """Get current tensor field."""
        return self.tensor_field.copy()
    
    def get_equilibrium_state(self) -> Dict:
        """Get equilibrium state information."""
        deviation = np.linalg.norm(self.target_field - self.tensor_field)
        is_equilibrium = deviation < 0.01
        
        return {
            'tensor_field': self.tensor_field.tolist(),
            'target_field': self.target_field.tolist(),
            'deviation': float(deviation),
            'is_equilibrium': is_equilibrium,
            'cycles': self.cycle_count,
            'field_norm': float(np.linalg.norm(self.tensor_field))
        }


class ResistanceComputer:
    """
    Computes resistance across the tensor resistor core.
    
    Calculates total, component, and effective resistance based on
    material properties and tensor field configuration.
    """
    
    def __init__(self, materials: List[ActiveMaterial]):
        """
        Initialize resistance computer.
        
        Args:
            materials: List of active materials in the core
        """
        self.materials = materials
        self.resistance_history = []
        self.computation_count = 0
    
    def compute_base_resistance(self, material: ActiveMaterial) -> float:
        """
        Compute base resistance for a material.
        
        Args:
            material: Material to compute resistance for
            
        Returns:
            Base resistance value
        """
        # Simplified resistance calculation: R = L / (σ * A)
        # Using normalized geometry
        conductivity = material.get_effective_conductivity()
        
        # Prevent division by zero
        if conductivity <= 0:
            return float('inf')
        
        base_resistance = 1.0 / conductivity
        return base_resistance
    
    def compute_total_resistance(self, tensor_field: np.ndarray) -> float:
        """
        Compute total system resistance.
        
        Args:
            tensor_field: Current tensor field from regulator
            
        Returns:
            Total resistance of the system
        """
        if not self.materials:
            return 0.0
        
        # Compute series resistance from all materials
        total_resistance = 0.0
        
        for material in self.materials:
            base_r = self.compute_base_resistance(material)
            
            # Tensor field modulation
            field_effect = 1.0 + 0.1 * np.linalg.norm(tensor_field)
            
            # Effective resistance
            effective_r = base_r * field_effect
            total_resistance += effective_r
        
        return total_resistance
    
    def compute_component_resistances(self) -> Dict[str, float]:
        """
        Compute resistance for each component.
        
        Returns:
            Dictionary of component resistances
        """
        resistances = {}
        
        for material in self.materials:
            resistance = self.compute_base_resistance(material)
            resistances[material.material_id] = resistance
        
        return resistances
    
    def compute_with_tensor_field(self, tensor_field: np.ndarray) -> Dict:
        """
        Compute complete resistance analysis with tensor field.
        
        Args:
            tensor_field: Current tensor field configuration
            
        Returns:
            Comprehensive resistance analysis
        """
        component_r = self.compute_component_resistances()
        total_r = self.compute_total_resistance(tensor_field)
        
        # Calculate parallel equivalent if treating as network
        if component_r:
            parallel_r = 1.0 / sum(1.0 / r for r in component_r.values() if r > 0)
        else:
            parallel_r = 0.0
        
        analysis = {
            'total_series': total_r,
            'total_parallel': parallel_r,
            'components': component_r,
            'tensor_influence': float(np.linalg.norm(tensor_field)),
            'computation_number': self.computation_count
        }
        
        self.resistance_history.append(analysis)
        self.computation_count += 1
        
        return analysis
    
    def get_resistance_statistics(self) -> Dict:
        """Get statistics from resistance history."""
        if not self.resistance_history:
            return {}
        
        series_values = [r['total_series'] for r in self.resistance_history]
        
        return {
            'computations': len(self.resistance_history),
            'avg_resistance': np.mean(series_values),
            'max_resistance': np.max(series_values),
            'min_resistance': np.min(series_values),
            'std_resistance': np.std(series_values)
        }


class EquilibriumSimulation:
    """
    Main simulation runner for the equilibrium resistance core.
    
    Orchestrates the interaction between active materials, tensor regulation,
    and resistance computation to simulate the complete system dynamics.
    """
    
    def __init__(self, num_materials: int = 3, simulation_id: str = "SIM_001"):
        """
        Initialize equilibrium simulation.
        
        Args:
            num_materials: Number of active materials to simulate
            simulation_id: Unique identifier for the simulation
        """
        self.simulation_id = simulation_id
        self.start_time = datetime.utcnow()
        
        # Initialize materials
        self.materials = self._create_materials(num_materials)
        
        # Initialize tensor regulator
        self.regulator = TensorRegulator(num_dimensions=3, regulation_strength=0.15)
        
        # Initialize resistance computer
        self.computer = ResistanceComputer(self.materials)
        
        # Simulation state
        self.step_count = 0
        self.max_steps = 100
        self.is_running = False
        self.simulation_log = []
    
    def _create_materials(self, count: int) -> List[ActiveMaterial]:
        """Create initial materials for the simulation."""
        materials = []
        
        for i in range(count):
            props = MaterialProperties(
                conductivity=1.0 + 0.5 * i,
                permeability=1.0,
                permittivity=1.0,
                temperature=300.0 + 50.0 * i
            )
            material = ActiveMaterial(f"MATERIAL_{i+1}", props)
            materials.append(material)
        
        return materials
    
    def setup_target_field(self, field_type: str = "identity"):
        """
        Setup target tensor field for regulation.
        
        Args:
            field_type: Type of field ("identity", "diagonal", "random")
        """
        if field_type == "identity":
            self.regulator.set_target_field(np.eye(3))
        elif field_type == "diagonal":
            self.regulator.set_target_field(np.diag([1.0, 0.5, 0.3]))
        elif field_type == "random":
            field = np.random.randn(3, 3) * 0.5
            self.regulator.set_target_field(field)
        else:
            raise ValueError(f"Unknown field type: {field_type}")
    
    def step(self) -> Dict:
        """
        Execute one simulation step.
        
        Returns:
            Step data dictionary
        """
        if not self.is_running:
            raise RuntimeError("Simulation not running. Call start() first.")
        
        # Simulate environmental changes
        for i, material in enumerate(self.materials):
            # Temperature oscillation
            temp = 300.0 + 50.0 * np.sin(self.step_count * 0.1 + i)
            material.update_temperature(temp, rate=0.1)
            
            # Periodic strain
            strain = 0.2 * np.sin(self.step_count * 0.2 + i * np.pi / 3)
            material.apply_strain(strain)
        
        # Regulate tensor field
        deviation = self.regulator.regulate_step()
        
        # Compute resistance
        tensor_field = self.regulator.get_tensor_field()
        resistance_data = self.computer.compute_with_tensor_field(tensor_field)
        
        # Create step record
        step_data = {
            'step': self.step_count,
            'timestamp': datetime.utcnow().isoformat(),
            'tensor_deviation': float(deviation),
            'total_resistance': resistance_data['total_series'],
            'material_states': [m.get_status() for m in self.materials],
            'equilibrium_state': self.regulator.get_equilibrium_state()
        }
        
        self.simulation_log.append(step_data)
        self.step_count += 1
        
        return step_data
    
    def start(self, max_steps: int = 100, target_field: str = "identity"):
        """
        Start the simulation.
        
        Args:
            max_steps: Maximum number of steps to run
            target_field: Type of target tensor field
        """
        self.max_steps = max_steps
        self.step_count = 0
        self.is_running = True
        self.setup_target_field(target_field)
    
    def run_full_simulation(self, max_steps: int = 100) -> List[Dict]:
        """
        Run complete simulation from start to finish.
        
        Args:
            max_steps: Maximum number of steps
            
        Returns:
            List of all step data
        """
        self.start(max_steps=max_steps)
        
        while self.step_count < self.max_steps:
            self.step()
        
        self.is_running = False
        return self.simulation_log
    
    def get_simulation_summary(self) -> Dict:
        """Get summary of simulation results."""
        if not self.simulation_log:
            return {}
        
        resistances = [s['total_resistance'] for s in self.simulation_log]
        deviations = [s['tensor_deviation'] for s in self.simulation_log]
        
        return {
            'simulation_id': self.simulation_id,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.utcnow().isoformat(),
            'total_steps': len(self.simulation_log),
            'materials': len(self.materials),
            'resistance_stats': {
                'final': resistances[-1] if resistances else 0.0,
                'average': float(np.mean(resistances)) if resistances else 0.0,
                'max': float(np.max(resistances)) if resistances else 0.0,
                'min': float(np.min(resistances)) if resistances else 0.0,
            },
            'equilibrium_stats': {
                'final_deviation': deviations[-1] if deviations else 0.0,
                'average_deviation': float(np.mean(deviations)) if deviations else 0.0,
            },
            'final_equilibrium': self.regulator.get_equilibrium_state()
        }
    
    def export_results(self, filename: str):
        """
        Export simulation results to JSON file.
        
        Args:
            filename: Output filename
        """
        results = {
            'summary': self.get_simulation_summary(),
            'full_log': self.simulation_log
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    """Main execution function demonstrating the complete system."""
    
    print("=" * 70)
    print("Equilibrium Resistance Core Simulation")
    print("=" * 70)
    print()
    
    # Create and run simulation
    print("Initializing simulation...")
    sim = EquilibriumSimulation(num_materials=3, simulation_id="ERC_2025_12_30")
    
    print("Running simulation...")
    results = sim.run_full_simulation(max_steps=50)
    
    # Display summary
    summary = sim.get_simulation_summary()
    
    print("\nSimulation Summary:")
    print("-" * 70)
    print(f"Simulation ID: {summary['simulation_id']}")
    print(f"Total Steps: {summary['total_steps']}")
    print(f"Materials: {summary['materials']}")
    print()
    
    print("Resistance Statistics:")
    print(f"  Final Resistance: {summary['resistance_stats']['final']:.6f} Ω")
    print(f"  Average Resistance: {summary['resistance_stats']['average']:.6f} Ω")
    print(f"  Max Resistance: {summary['resistance_stats']['max']:.6f} Ω")
    print(f"  Min Resistance: {summary['resistance_stats']['min']:.6f} Ω")
    print()
    
    print("Equilibrium Statistics:")
    print(f"  Final Deviation: {summary['equilibrium_stats']['final_deviation']:.6f}")
    print(f"  Average Deviation: {summary['equilibrium_stats']['average_deviation']:.6f}")
    print(f"  Equilibrium Reached: {summary['final_equilibrium']['is_equilibrium']}")
    print()
    
    print("Final Material States:")
    final_step = results[-1]
    for mat_state in final_step['material_states']:
        print(f"  {mat_state['material_id']}:")
        print(f"    Temperature: {mat_state['properties']['temperature']:.2f} K")
        print(f"    Conductivity: {mat_state['effective_conductivity']:.6f}")
        print()
    
    # Export results
    print("Exporting results to 'equilibrium_simulation_results.json'...")
    sim.export_results('equilibrium_simulation_results.json')
    print("Export complete!")
    
    print("\n" + "=" * 70)
    print("Simulation completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
