"""
TRC-132 Traffic MVP Implementation
Complete implementation of the Tensor-Resistor-Core Traffic Management System
with Modal Logic Engine, Lane Topology (|1|3|2|), and Pressure Discharge Calculation

Author: David Besemer
Date: 2025-12-30
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class LaneType(Enum):
    """Lane type enumeration for the traffic topology"""
    ENTRY = 1
    MAIN = 3
    EXIT = 2


@dataclass
class TrafficState:
    """Represents the state of traffic at a given moment"""
    timestamp: float
    vehicle_count: int
    flow_rate: float  # vehicles per unit time
    congestion_level: float  # 0.0 to 1.0
    pressure: float  # karmic debt accumulation


@dataclass
class KarmicDebtRecord:
    """Records karmic debt incidents"""
    timestamp: float
    incident_type: str  # 'delay', 'congestion', 'incident'
    magnitude: float
    affected_vehicles: int
    
    def calculate_debt_impact(self) -> float:
        """Calculate the impact of this incident on karmic debt"""
        return self.magnitude * self.affected_vehicles


@dataclass
class ModalTransition:
    """Represents a transition between modal states"""
    from_mode: str
    to_mode: str
    trigger_condition: str
    timestamp: float
    confidence: float = 1.0


class ModalLogicEngine:
    """
    Modal Logic Engine for traffic state management
    Implements possibility and necessity operators for traffic flow analysis
    """
    
    MODES = {
        'free_flow': {
            'description': 'Free flowing traffic',
            'threshold': 0.3,
            'discharge_rate': 0.05,
        },
        'moderate': {
            'description': 'Moderate traffic density',
            'threshold': 0.6,
            'discharge_rate': 0.02,
        },
        'congested': {
            'description': 'Heavy congestion',
            'threshold': 0.85,
            'discharge_rate': 0.001,
        },
        'critical': {
            'description': 'Critical gridlock',
            'threshold': 1.0,
            'discharge_rate': 0.0001,
        }
    }
    
    def __init__(self):
        self.current_mode = 'free_flow'
        self.mode_history: List[ModalTransition] = []
        self.mode_durations: Dict[str, float] = defaultdict(float)
    
    def evaluate_mode(self, congestion_level: float) -> str:
        """
        Determine the current modal state based on congestion level
        Applies possibility (◇) and necessity (□) operators
        """
        sorted_modes = sorted(
            self.MODES.items(),
            key=lambda x: x[1]['threshold']
        )
        
        for mode_name, mode_config in sorted_modes:
            if congestion_level <= mode_config['threshold']:
                return mode_name
        
        return 'critical'
    
    def possibility_operator(self, congestion_level: float, mode: str) -> float:
        """
        Possibility operator (◇): Can the system reach this mode?
        Returns probability of reaching this mode
        """
        threshold = self.MODES[mode]['threshold']
        if congestion_level <= threshold:
            return min(1.0, congestion_level / threshold + 0.1)
        return max(0.0, 1.0 - (congestion_level - threshold) / (1.0 - threshold))
    
    def necessity_operator(self, congestion_level: float, mode: str) -> float:
        """
        Necessity operator (□): Must the system be in this mode?
        Returns confidence that system must be in this mode
        """
        current_threshold = self.MODES[mode]['threshold']
        
        if mode == 'critical':
            return float(congestion_level > 0.85)
        
        mode_list = list(self.MODES.keys())
        current_idx = mode_list.index(mode)
        next_threshold = self.MODES[mode_list[current_idx + 1]]['threshold']
        
        if current_threshold <= congestion_level <= next_threshold:
            return (congestion_level - current_threshold) / (next_threshold - current_threshold)
        
        return 0.0
    
    def transition_to_mode(self, new_mode: str, trigger: str, timestamp: float) -> bool:
        """Transition to a new modal state"""
        if new_mode in self.MODES and new_mode != self.current_mode:
            transition = ModalTransition(
                from_mode=self.current_mode,
                to_mode=new_mode,
                trigger_condition=trigger,
                timestamp=timestamp,
                confidence=self.necessity_operator(0.5, new_mode)  # Placeholder confidence
            )
            self.mode_history.append(transition)
            self.current_mode = new_mode
            return True
        return False
    
    def get_discharge_rate(self) -> float:
        """Get the pressure discharge rate for current mode"""
        return self.MODES[self.current_mode]['discharge_rate']


class LaneTopology:
    """
    Lane Topology Manager implementing |1|3|2| configuration
    Entry Lane (1) -> Main Lane (3) -> Exit Lane (2)
    """
    
    def __init__(self):
        self.lanes: Dict[LaneType, 'Lane'] = {
            LaneType.ENTRY: Lane(LaneType.ENTRY, capacity=50),
            LaneType.MAIN: Lane(LaneType.MAIN, capacity=100),
            LaneType.EXIT: Lane(LaneType.EXIT, capacity=50),
        }
        self.flow_distribution = {
            'entry_to_main': 0.9,  # 90% flow from entry to main
            'main_to_exit': 0.85,  # 85% flow from main to exit
        }
    
    def get_topology_string(self) -> str:
        """Return the lane topology visualization"""
        return "|1|3|2|"
    
    def calculate_flow_through_lanes(self, entry_flow: float) -> Dict[LaneType, float]:
        """Calculate flow through each lane"""
        main_flow = entry_flow * self.flow_distribution['entry_to_main']
        exit_flow = main_flow * self.flow_distribution['main_to_exit']
        
        return {
            LaneType.ENTRY: entry_flow,
            LaneType.MAIN: main_flow,
            LaneType.EXIT: exit_flow,
        }
    
    def get_congestion_by_lane(self) -> Dict[str, float]:
        """Get congestion level for each lane"""
        return {
            'entry': self.lanes[LaneType.ENTRY].get_congestion(),
            'main': self.lanes[LaneType.MAIN].get_congestion(),
            'exit': self.lanes[LaneType.EXIT].get_congestion(),
        }
    
    def update_lane_vehicles(self, flows: Dict[LaneType, float]):
        """Update vehicle count in each lane"""
        for lane_type, flow in flows.items():
            self.lanes[lane_type].add_vehicles(int(flow))


class Lane:
    """Represents a single traffic lane"""
    
    def __init__(self, lane_type: LaneType, capacity: int):
        self.lane_type = lane_type
        self.capacity = capacity
        self.vehicle_count = 0
    
    def add_vehicles(self, count: int):
        """Add vehicles to the lane (capped at capacity)"""
        self.vehicle_count = min(self.vehicle_count + count, self.capacity)
    
    def remove_vehicles(self, count: int):
        """Remove vehicles from the lane"""
        self.vehicle_count = max(self.vehicle_count - count, 0)
    
    def get_congestion(self) -> float:
        """Get congestion level (0.0 to 1.0)"""
        return self.vehicle_count / self.capacity


class PressureCalculationEngine:
    """
    Pressure Discharge Calculation Engine
    Implements the mathematical framework for karmic debt tracking and discharge
    """
    
    def __init__(self):
        self.total_pressure: float = 0.0
        self.karmic_debt_records: List[KarmicDebtRecord] = []
        self.discharge_history: List[Tuple[float, float]] = []  # (timestamp, amount_discharged)
        self.pressure_history: List[Tuple[float, float]] = []  # (timestamp, pressure_level)
    
    def add_karmic_debt(self, record: KarmicDebtRecord):
        """Add a karmic debt incident"""
        self.karmic_debt_records.append(record)
        debt_impact = record.calculate_debt_impact()
        self.total_pressure += debt_impact
    
    def calculate_pressure(self, 
                          modal_logic_engine: ModalLogicEngine,
                          lane_congestion: Dict[str, float],
                          timestamp: float) -> float:
        """
        Calculate current pressure (karmic debt) in the system
        
        Pressure = Base Karmic Debt + (Lane Congestion Factor) - (Discharge Rate Effect)
        """
        # Base pressure from karmic debt records
        base_pressure = sum(record.calculate_debt_impact() 
                           for record in self.karmic_debt_records)
        
        # Congestion contribution to pressure
        avg_congestion = np.mean(list(lane_congestion.values()))
        congestion_pressure = avg_congestion * 100  # Scale factor
        
        # Discharge rate reduces pressure over time
        discharge_rate = modal_logic_engine.get_discharge_rate()
        time_factor = discharge_rate * len(self.discharge_history)
        
        # Calculate final pressure
        total_pressure = base_pressure + congestion_pressure - (time_factor * 10)
        self.total_pressure = max(0.0, total_pressure)
        
        self.pressure_history.append((timestamp, self.total_pressure))
        return self.total_pressure
    
    def discharge_pressure(self, 
                          discharge_amount: float, 
                          timestamp: float,
                          reason: str = "normal") -> float:
        """
        Discharge pressure from the system
        
        Proof of Pressure Discharge:
        If P(t) = current pressure at time t
        And D(r) = discharge amount based on rate r
        Then P(t+1) = max(0, P(t) - D(r))
        
        This ensures monotonic decrease in pressure when discharge is active
        """
        if self.total_pressure <= 0:
            return 0.0
        
        actual_discharge = min(discharge_amount, self.total_pressure)
        self.total_pressure -= actual_discharge
        self.discharge_history.append((timestamp, actual_discharge))
        
        return actual_discharge
    
    def get_pressure_metrics(self) -> Dict:
        """Get comprehensive pressure metrics"""
        total_discharged = sum(amount for _, amount in self.discharge_history)
        total_accumulated = sum(record.calculate_debt_impact() 
                               for record in self.karmic_debt_records)
        
        return {
            'current_pressure': self.total_pressure,
            'total_accumulated': total_accumulated,
            'total_discharged': total_discharged,
            'net_pressure': total_accumulated - total_discharged,
            'discharge_count': len(self.discharge_history),
            'karmic_debt_incidents': len(self.karmic_debt_records),
        }


class Conductor:
    """
    Traffic Conductor: Orchestrates the entire Traffic MVP system
    Coordinates Modal Logic Engine, Lane Topology, and Pressure Calculation
    """
    
    def __init__(self, simulation_id: str = "TRC-132-DEFAULT"):
        self.simulation_id = simulation_id
        self.modal_engine = ModalLogicEngine()
        self.lane_topology = LaneTopology()
        self.pressure_engine = PressureCalculationEngine()
        self.traffic_states: List[TrafficState] = []
        self.current_time = 0.0
        self.is_running = False
    
    def initialize_system(self):
        """Initialize the traffic management system"""
        self.is_running = True
        print(f"[CONDUCTOR] Initializing TRC-132 Traffic MVP: {self.simulation_id}")
        print(f"[CONDUCTOR] Lane Topology: {self.lane_topology.get_topology_string()}")
        print(f"[CONDUCTOR] Modal Logic Engine: Online")
        print(f"[CONDUCTOR] Pressure Calculation Engine: Online")
    
    def step_simulation(self, 
                       entry_flow: float,
                       incident_records: Optional[List[KarmicDebtRecord]] = None,
                       delta_time: float = 1.0):
        """
        Execute one simulation step
        
        Args:
            entry_flow: Number of vehicles entering the system
            incident_records: List of karmic debt incidents occurring this step
            delta_time: Time delta for this step
        """
        self.current_time += delta_time
        
        # Calculate flow through lanes
        flows = self.lane_topology.calculate_flow_through_lanes(entry_flow)
        self.lane_topology.update_lane_vehicles(flows)
        
        # Get current congestion
        lane_congestion = self.lane_topology.get_congestion_by_lane()
        avg_congestion = np.mean(list(lane_congestion.values()))
        
        # Update modal state
        new_mode = self.modal_engine.evaluate_mode(avg_congestion)
        if new_mode != self.modal_engine.current_mode:
            self.modal_engine.transition_to_mode(
                new_mode, 
                f"Congestion: {avg_congestion:.2f}",
                self.current_time
            )
        
        # Process karmic debt incidents
        if incident_records:
            for record in incident_records:
                self.pressure_engine.add_karmic_debt(record)
        
        # Calculate pressure
        pressure = self.pressure_engine.calculate_pressure(
            self.modal_engine,
            lane_congestion,
            self.current_time
        )
        
        # Discharge pressure based on modal state
        discharge_rate = self.modal_engine.get_discharge_rate()
        discharge_amount = pressure * discharge_rate
        self.pressure_engine.discharge_pressure(
            discharge_amount,
            self.current_time,
            reason=f"Mode: {self.modal_engine.current_mode}"
        )
        
        # Record traffic state
        state = TrafficState(
            timestamp=self.current_time,
            vehicle_count=sum(flows.values()),
            flow_rate=entry_flow,
            congestion_level=avg_congestion,
            pressure=pressure
        )
        self.traffic_states.append(state)
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'simulation_id': self.simulation_id,
            'current_time': self.current_time,
            'is_running': self.is_running,
            'modal_state': self.modal_engine.current_mode,
            'lane_topology': self.lane_topology.get_topology_string(),
            'lane_congestion': self.lane_topology.get_congestion_by_lane(),
            'pressure_metrics': self.pressure_engine.get_pressure_metrics(),
            'mode_history_count': len(self.modal_engine.mode_history),
        }
    
    def run_simulation(self, 
                      duration: float,
                      flow_profile: callable,
                      incident_generator: Optional[callable] = None):
        """
        Run a complete simulation
        
        Args:
            duration: Total simulation time
            flow_profile: Function that returns entry flow for given time
            incident_generator: Function that returns incidents for given time
        """
        self.initialize_system()
        
        steps = int(duration)
        for step in range(steps):
            current_time = float(step)
            entry_flow = flow_profile(current_time)
            incidents = incident_generator(current_time) if incident_generator else None
            
            self.step_simulation(entry_flow, incidents, delta_time=1.0)
        
        print(f"\n[CONDUCTOR] Simulation Complete")
        self._print_summary()
    
    def _print_summary(self):
        """Print simulation summary"""
        status = self.get_system_status()
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        print(f"Simulation ID: {status['simulation_id']}")
        print(f"Duration: {status['current_time']} time units")
        print(f"Lane Topology: {status['lane_topology']}")
        print(f"Final Modal State: {status['modal_state']}")
        print(f"Modal State Transitions: {status['mode_history_count']}")
        
        print("\nLane Congestion Levels:")
        for lane, congestion in status['lane_congestion'].items():
            print(f"  {lane.upper()}: {congestion:.2%}")
        
        pressure = status['pressure_metrics']
        print("\nPressure Metrics:")
        print(f"  Current Pressure: {pressure['current_pressure']:.2f}")
        print(f"  Total Accumulated: {pressure['total_accumulated']:.2f}")
        print(f"  Total Discharged: {pressure['total_discharged']:.2f}")
        print(f"  Net Pressure: {pressure['net_pressure']:.2f}")
        print(f"  Discharge Events: {pressure['discharge_count']}")
        print(f"  Karmic Incidents: {pressure['karmic_debt_incidents']}")
        print("="*60 + "\n")
    
    def export_simulation_data(self) -> Dict:
        """Export complete simulation data"""
        return {
            'metadata': {
                'simulation_id': self.simulation_id,
                'total_duration': self.current_time,
                'timestamp': '2025-12-30T16:01:41Z',
            },
            'modal_logic': {
                'final_mode': self.modal_engine.current_mode,
                'transitions': [
                    {
                        'from': t.from_mode,
                        'to': t.to_mode,
                        'trigger': t.trigger_condition,
                        'time': t.timestamp,
                        'confidence': t.confidence,
                    }
                    for t in self.modal_engine.mode_history
                ],
            },
            'lane_topology': {
                'configuration': self.lane_topology.get_topology_string(),
                'final_congestion': self.lane_topology.get_congestion_by_lane(),
            },
            'pressure_data': self.pressure_engine.get_pressure_metrics(),
            'traffic_states': [
                {
                    'timestamp': s.timestamp,
                    'vehicle_count': s.vehicle_count,
                    'flow_rate': s.flow_rate,
                    'congestion': s.congestion_level,
                    'pressure': s.pressure,
                }
                for s in self.traffic_states[-10:]  # Last 10 states
            ],
        }


# Example simulation functions
def example_flow_profile(time: float) -> float:
    """Example traffic flow profile"""
    # Simulate rush hour effect
    if 7 <= (time % 24) < 9 or 17 <= (time % 24) < 19:
        return 40.0 + 20.0 * np.sin(time / 3)
    else:
        return 20.0 + 10.0 * np.sin(time / 5)


def example_incident_generator(time: float) -> Optional[List[KarmicDebtRecord]]:
    """Example incident generator"""
    incidents = []
    
    # Occasional minor incidents
    if int(time) % 10 == 5 and time > 5:
        incidents.append(KarmicDebtRecord(
            timestamp=time,
            incident_type='delay',
            magnitude=5.0,
            affected_vehicles=15,
        ))
    
    # Occasional major incidents
    if int(time) % 30 == 15 and time > 15:
        incidents.append(KarmicDebtRecord(
            timestamp=time,
            incident_type='congestion',
            magnitude=10.0,
            affected_vehicles=50,
        ))
    
    return incidents if incidents else None


if __name__ == "__main__":
    """
    Run the TRC-132 Traffic MVP demonstration
    """
    print("\n" + "="*60)
    print("TRC-132 TRAFFIC MVP - TENSOR RESISTOR CORE")
    print("Modal Logic | Lane Topology (|1|3|2|) | Pressure Calculation")
    print("="*60 + "\n")
    
    # Create and run conductor
    conductor = Conductor(simulation_id="TRC-132-DEMO-001")
    
    # Run simulation for 100 time units
    conductor.run_simulation(
        duration=100,
        flow_profile=example_flow_profile,
        incident_generator=example_incident_generator,
    )
    
    # Export simulation data
    export_data = conductor.export_simulation_data()
    print("\nSimulation data exported successfully.")
    print(f"Modal transitions recorded: {len(export_data['modal_logic']['transitions'])}")
    print(f"Final pressure state: {export_data['pressure_data']['current_pressure']:.2f}")
