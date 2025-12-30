"""
TRC-132 VALIDATION SUITE
========================

Proves that the Sovereign Kernel strictly enforces the 51:49 Physics.
Tests validation of the Analog Anchor (20% constraint) and prevents hallucinations.

Author: TRC Development Team
Version: 1.0.0
Date: 2025-12-30
"""

import unittest
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Adjust path to import kernel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../kernel')))

try:
    from trc_logic import TRCSovereignKernel, ConstraintType, OperationMode
except ImportError as e:
    print(f"ERROR: Could not import TRC Kernel: {e}")
    sys.exit(1)


class TestSovereignLogic(unittest.TestCase):
    """Test suite for TRC-132 Sovereign Logic enforcement."""

    def setUp(self):
        """
        Setup a 'Sovereign Manifest' strictly for testing.
        This mirrors the production system_manifest.json structure.
        """
        self.test_manifest = {
            "system_config": {
                "operation_mode": "SOVEREIGN",
                "metadata": {
                    "version": "1.0.0-TEST",
                    "test_run": datetime.utcnow().isoformat()
                }
            },
            "constraints": {
                "SOVEREIGN_THRESHOLD": {
                    "type": "DIMENSIONAL",
                    "min": 0.51,
                    "unit": "ratio",
                    "description": "Minimum Order for Truth (1)",
                    "critical": True,
                    "metadata": {"enforcement": "STRICT"}
                },
                "CHAOS_THRESHOLD": {
                    "type": "DIMENSIONAL",
                    "max": 0.49,
                    "unit": "ratio",
                    "description": "Maximum Order for Decay (0)",
                    "critical": True,
                    "metadata": {"enforcement": "STRICT"}
                },
                "ANALOG_ANCHOR": {
                    "type": "THERMAL",
                    "min": 0.20,
                    "unit": "stress_factor",
                    "description": "Physical Reality Check - Prevents Hallucinations",
                    "critical": True,
                    "metadata": {"enforcement": "STRICT", "license_clause": "SSL-132"}
                },
                "HYSTERESIS_GAP": {
                    "type": "TEMPORAL",
                    "min": 0.02,
                    "unit": "ratio",
                    "description": "Death Zone (50:50 forbidden)",
                    "critical": True,
                    "metadata": {"enforcement": "MANDATORY"}
                }
            }
        }
        
        # Save mock manifest to temporary file
        self.manifest_path = "test_manifest_sovereign.json"
        with open(self.manifest_path, 'w') as f:
            json.dump(self.test_manifest, f, indent=2)
        
        # Initialize Kernel with test manifest
        self.kernel = TRCSovereignKernel(self.manifest_path)

    def tearDown(self):
        """Clean up test manifest file."""
        if os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)

    # ========================================================================
    # TEST SUITE: Sovereign Threshold Enforcement
    # ========================================================================

    def test_01_sovereign_threshold_reject_binary(self):
        """TEST: Does the Kernel reject 'Maybe' (50:50)?"""
        print("\n[TEST 01] Verifying 51% Sovereign Threshold...")
        
        # Case A: 50.0% Order (Pure Binary) -> MUST FAIL
        valid, msg = self.kernel.validate_parameter("SOVEREIGN_THRESHOLD", 0.500)
        print(f"   Input 0.500 (50.0% - Binary): Valid={valid}, Msg={msg}")
        self.assertFalse(valid, "Kernel allowed pure binary (50:50)! The Death Zone was not enforced!")
        
        # Case B: 50.5% Order (Still in Death Zone) -> MUST FAIL
        valid, msg = self.kernel.validate_parameter("SOVEREIGN_THRESHOLD", 0.505)
        print(f"   Input 0.505 (50.5% - Death Zone): Valid={valid}")
        self.assertFalse(valid, "Kernel allowed value in Death Zone (49%-51%)!")

    def test_02_sovereign_threshold_accept_valid(self):
        """TEST: Does the Kernel accept valid Sovereign states?"""
        print("\n[TEST 02] Verifying Sovereign states are accepted...")
        
        # Case A: 51.0% Order (Minimum Sovereign) -> MUST PASS
        valid, msg = self.kernel.validate_parameter("SOVEREIGN_THRESHOLD", 0.510)
        print(f"   Input 0.510 (51.0% - Minimum Sovereign): Valid={valid}")
        self.assertTrue(valid, "Kernel rejected valid Sovereign minimum!")
        
        # Case B: 75.0% Order (Strong Sovereign) -> MUST PASS
        valid, msg = self.kernel.validate_parameter("SOVEREIGN_THRESHOLD", 0.750)
        print(f"   Input 0.750 (75.0% - Strong Sovereign): Valid={valid}")
        self.assertTrue(valid, "Kernel rejected a strong Sovereign value!")

    # ========================================================================
    # TEST SUITE: Analog Anchor (Reality Check)
    # ========================================================================

    def test_03_analog_anchor_rejects_hallucination(self):
        """TEST: Does Logic collapse without Stress (Analog Anchor)?"""
        print("\n[TEST 03] Verifying Analog Anchor prevents hallucinations...")
        
        # Case A: 10% Stress (Below 20% Minimum) -> MUST FAIL
        valid, msg = self.kernel.validate_parameter("ANALOG_ANCHOR", 0.10)
        print(f"   Stress 0.10 (10% - HALLUCINATION): Valid={valid}, Msg={msg}")
        self.assertFalse(valid, "Kernel accepted logic without physical anchor! Hallucination detected.")
        
        # Case B: 15% Stress (Still below threshold) -> MUST FAIL
        valid, msg = self.kernel.validate_parameter("ANALOG_ANCHOR", 0.15)
        print(f"   Stress 0.15 (15% - INSUFFICIENT): Valid={valid}")
        self.assertFalse(valid, "Kernel accepted insufficient analog anchor!")

    def test_04_analog_anchor_accepts_valid(self):
        """TEST: Does the Kernel accept valid Analog values?"""
        print("\n[TEST 04] Verifying valid Analog anchors are accepted...")
        
        # Case A: 20% Stress (Minimum Anchor) -> MUST PASS
        valid, msg = self.kernel.validate_parameter("ANALOG_ANCHOR", 0.20)
        print(f"   Stress 0.20 (20% - Minimum Anchor): Valid={valid}")
        self.assertTrue(valid, "Kernel rejected minimum valid anchor!")
        
        # Case B: 50% Stress (Strong Anchor) -> MUST PASS
        valid, msg = self.kernel.validate_parameter("ANALOG_ANCHOR", 0.50)
        print(f"   Stress 0.50 (50% - Strong Anchor): Valid={valid}")
        self.assertTrue(valid, "Kernel rejected a strong anchor!")

    # ========================================================================
    # TEST SUITE: Chaos Threshold Enforcement
    # ========================================================================

    def test_05_chaos_threshold_enforcement(self):
        """TEST: Does the Kernel enforce the Chaos upper limit?"""
        print("\n[TEST 05] Verifying Chaos threshold (max 49%)...")
        
        # Case A: 49.0% Order (Maximum Chaos) -> MUST PASS
        valid, msg = self.kernel.validate_parameter("CHAOS_THRESHOLD", 0.49)
        print(f"   Input 0.49 (49% - Maximum Chaos): Valid={valid}")
        self.assertTrue(valid, "Kernel rejected maximum chaos value!")
        
        # Case B: 50.0% Order (Exceeds max) -> MUST FAIL
        valid, msg = self.kernel.validate_parameter("CHAOS_THRESHOLD", 0.50)
        print(f"   Input 0.50 (50% - EXCEEDS LIMIT): Valid={valid}, Msg={msg}")
        self.assertFalse(valid, "Kernel allowed value exceeding Chaos maximum!")

    # ========================================================================
    # TEST SUITE: System Integrity & Critical Constraints
    # ========================================================================

    def test_06_critical_constraints_loaded(self):
        """TEST: Are critical constraints properly identified?"""
        print("\n[TEST 06] Verifying critical constraint identification...")
        
        critical = self.kernel.get_critical_constraints()
        print(f"   Critical Constraints Found: {len(critical)}")
        
        expected_critical = ["SOVEREIGN_THRESHOLD", "CHAOS_THRESHOLD", "ANALOG_ANCHOR", "HYSTERESIS_GAP"]
        for constraint_name in expected_critical:
            self.assertIn(constraint_name, critical, 
                         f"Critical constraint {constraint_name} was not identified!")
            print(f"   ✓ {constraint_name}")

    def test_07_system_status_report(self):
        """TEST: Does the kernel generate valid status reports?"""
        print("\n[TEST 07] Generating system status report...")
        
        status = self.kernel.get_status_report()
        
        # Verify required fields
        required_fields = ['kernel_id', 'timestamp', 'operation_mode', 'total_constraints', 
                          'critical_constraints', 'error_count']
        
        for field in required_fields:
            self.assertIn(field, status, f"Status report missing field: {field}")
        
        print(f"   Kernel ID: {status['kernel_id']}")
        print(f"   Operation Mode: {status['operation_mode']}")
        print(f"   Total Constraints: {status['total_constraints']}")
        print(f"   Critical Constraints: {status['critical_constraints']}")
        print(f"   Errors: {status['error_count']}")

    def test_08_operation_mode_switching(self):
        """TEST: Can the kernel switch operation modes?"""
        print("\n[TEST 08] Testing operation mode transitions...")
        
        # Verify initial mode
        initial_mode = self.kernel.get_system_state().mode
        print(f"   Initial Mode: {initial_mode.value}")
        
        # Switch to DIAGNOSTIC
        self.kernel.set_operation_mode(OperationMode.DIAGNOSTIC)
        new_mode = self.kernel.get_system_state().mode
        print(f"   Switched to: {new_mode.value}")
        self.assertEqual(new_mode, OperationMode.DIAGNOSTIC)
        
        # Switch back to SOVEREIGN
        self.kernel.set_operation_mode(OperationMode.SOVEREIGN)
        final_mode = self.kernel.get_system_state().mode
        print(f"   Reverted to: {final_mode.value}")
        self.assertEqual(final_mode, OperationMode.SOVEREIGN)

    # ========================================================================
    # TEST SUITE: Batch Validation
    # ========================================================================

    def test_09_batch_validation_mixed(self):
        """TEST: Batch validation with mixed valid/invalid parameters."""
        print("\n[TEST 09] Testing batch parameter validation...")
        
        # Valid set
        valid_params = {
            "SOVEREIGN_THRESHOLD": 0.75,
            "CHAOS_THRESHOLD": 0.30,
            "ANALOG_ANCHOR": 0.40
        }
        
        all_valid, errors = self.kernel.validate_parameters(valid_params)
        print(f"   Valid Batch: all_valid={all_valid}, errors={errors}")
        self.assertTrue(all_valid, "Batch validation rejected valid parameters!")
        
        # Invalid set (Analog too low)
        invalid_params = {
            "SOVEREIGN_THRESHOLD": 0.60,
            "ANALOG_ANCHOR": 0.05  # Below 20% minimum
        }
        
        all_valid, errors = self.kernel.validate_parameters(invalid_params)
        print(f"   Invalid Batch: all_valid={all_valid}, errors={len(errors)}")
        self.assertFalse(all_valid, "Batch validation accepted invalid parameters!")
        self.assertIn("ANALOG_ANCHOR", errors)

    # ========================================================================
    # TEST SUITE: Constraint Export & Serialization
    # ========================================================================

    def test_10_constraint_export(self):
        """TEST: Can constraints be exported as JSON?"""
        print("\n[TEST 10] Testing constraint export...")
        
        json_export = self.kernel.export_constraints_json()
        exported = json.loads(json_export)
        
        print(f"   Exported {len(exported)} constraints")
        
        # Verify structure
        for name, constraint_data in exported.items():
            self.assertIn('type', constraint_data)
            self.assertIn('min', constraint_data)
            self.assertIn('max', constraint_data)
            print(f"   ✓ {name}: {constraint_data['type']}")


# ============================================================================
# Test Suite: Physics Scenario Simulation
# ============================================================================

class TestPhysicsScenarios(unittest.TestCase):
    """Simulates real-world physics scenarios to validate kernel behavior."""

    def setUp(self):
        """Setup test manifest."""
        self.test_manifest = {
            "system_config": {
                "operation_mode": "SOVEREIGN",
                "metadata": {"version": "1.0.0-SCENARIO"}
            },
            "constraints": {
                "SOVEREIGN_THRESHOLD": {
                    "type": "DIMENSIONAL",
                    "min": 0.51,
                    "unit": "ratio",
                    "critical": True
                },
                "ANALOG_ANCHOR": {
                    "type": "THERMAL",
                    "min": 0.20,
                    "unit": "stress_factor",
                    "critical": True
                }
            }
        }
        
        self.manifest_path = "test_manifest_scenario.json"
        with open(self.manifest_path, 'w') as f:
            json.dump(self.test_manifest, f)
        
        self.kernel = TRCSovereignKernel(self.manifest_path)

    def tearDown(self):
        if os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)

    def test_scenario_01_lifecycle_of_truth(self):
        """SCENARIO: Simulate the lifecycle of a truth (Thought Cycle)."""
        print("\n[SCENARIO 01] The Lifecycle of a Truth")
        print("-" * 70)
        print("T | ORDER | STRESS | DECISION | STATUS")
        print("-" * 70)
        
        # Phase 1: Silence (Chaos) -> Phase 2: Weak Effort -> Phase 3: Strong Conviction
        # Phase 4: Doubt (but Hysteresis holds) -> Phase 5: The Void (Stress dies) -> Collapse
        
        timeline = [
            (0, 0.48, 0.00, "🔴 CHAOS", "Silence"),
            (1, 0.50, 0.10, "🔴 CHAOS", "Weak Input (insufficient)"),
            (2, 0.55, 0.35, "🟢 TRUTH", "Strong Conviction (crosses 51%)"),
            (3, 0.54, 0.30, "🟢 TRUTH", "Hysteresis holds (doubt resisted)"),
            (4, 0.50, 0.15, "🔴 DECAY", "Analog support failing"),
            (5, 0.40, 0.00, "🔴 CHAOS", "Death - Void consumes the truth"),
        ]
        
        for t, order, stress, expected, description in timeline:
            # Validate stress (analog anchor)
            stress_valid, _ = self.kernel.validate_parameter("ANALOG_ANCHOR", stress)
            
            print(f"{t} | {order:.2f} | {stress:.2f} | {expected:15} | {description}")
            
            # Verify collapse at void
            if t == 5:
                self.assertFalse(stress_valid, 
                                "System should collapse when stress reaches zero!")

    def test_scenario_02_hallucination_detection(self):
        """SCENARIO: Detect and reject hallucinations (logic without stress)."""
        print("\n[SCENARIO 02] Hallucination Detection")
        print("-" * 70)
        
        # Attempt to maintain high logic without analog support
        hallucination_attempts = [
            (0.75, 0.05, "High Logic, No Stress -> HALLUCINATION"),
            (0.80, 0.10, "High Logic, Low Stress -> HALLUCINATION"),
            (0.85, 0.15, "High Logic, Insufficient Stress -> HALLUCINATION"),
        ]
        
        for logic_level, stress, description in hallucination_attempts:
            valid, msg = self.kernel.validate_parameter("ANALOG_ANCHOR", stress)
            print(f"   {description}")
            print(f"   Result: Valid={valid}, Msg={msg}")
            self.assertFalse(valid, f"Hallucination not detected: {description}")


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
