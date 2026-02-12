"""
Resource management interfaces for scheduling, power control, and pilot reuse.

This module provides interfaces and implementations for dynamic resource
management in MIMO-OFDM systems, including user scheduling, power control,
and pilot reuse strategies. Resource management is essential for optimizing
system performance, managing interference, and adapting to channel conditions.

Theory:
    Resource Management in Wireless Systems:
    
    1. User Scheduling:
       - Select which users to serve in each time slot
       - Maximize throughput: max Σ R_i subject to constraints
       - Proportional fairness: max Σ log(R_i) for fairness
       - Round-robin: Serve users in rotation
       - Channel-aware: Prefer users with good channel conditions
       
    2. Power Control:
       - Adjust transmit power per user to optimize performance
       - Objectives:
         * Maximize sum rate: max Σ log(1 + SNR_i)
         * Minimize interference: min Σ I_i
         * Equalize SINR: SNR_i ≈ SNR_target for all i
       - Water-filling: Allocate more power to better channels
       - Fractional power control: P_i ∝ (d_i/d_0)^(-α·β)
         where β is path loss compensation factor
       
    3. Pilot Reuse:
       - Reuse pilots across cells to save resources
       - Pilot contamination: Interference from other cells using same pilots
       - Reuse factor K: 1/K of cells use same pilots
       - Higher K: Less contamination, more pilot overhead
       - Lower K: More contamination, less pilot overhead
       
    4. Link Adaptation:
       - Adjust modulation and coding scheme (MCS) based on channel
       - Select MCS to achieve target BLER (e.g., 1e-3)
       - Adaptive modulation: Higher order for good channels
       - Adaptive coding: Lower code rate for poor channels
       
    5. Interference Management:
       - Coordinate resource allocation across cells
       - Inter-cell interference coordination (ICIC)
       - Coordinated multipoint (CoMP) transmission
       - Beamforming to null interference

References:
    - Tse & Viswanath, "Fundamentals of Wireless Communication" (Resource allocation)
    - Björnson et al., "Massive MIMO Networks" (Power control, scheduling)
    - 3GPP TS 38.214: Physical layer procedures for data
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any




@dataclass
class ResourceDirectives:
    """
    Runtime resource directives applied per batch.
    
    This dataclass encapsulates resource management decisions that are applied
    dynamically during simulation. The directives control user scheduling,
    power allocation, and pilot reuse on a per-batch basis.
    
    Attributes:
        active_ut_mask: Binary mask indicating which UTs are scheduled.
            Length = num_ut, values 0 or 1. 1 = active/scheduled, 0 = muted.
            Used for dynamic scheduling and interference management.
            
        per_ut_power: Per-UT power scaling factors (linear scale).
            Length = num_ut. Used for power control to balance received signal
            strengths and manage interference. Values typically in range [0, 1].
            Applied as: x_scaled = x * sqrt(power_factor).
            
        pilot_reuse_factor: Pilot reuse factor for interference management.
            Reusing pilots across cells causes pilot contamination. Higher reuse
            reduces contamination but requires more orthogonal pilot sequences.
            Factor of 1 means no reuse (all cells use same pilots).
    """
    active_ut_mask: Optional[List[int]] = None
    per_ut_power: Optional[List[float]] = None
    # Placeholder for future pilot reuse / pattern changes
    pilot_reuse_factor: Optional[int] = None


class ResourceManager:
    """
    Base interface for resource management.
    
    This abstract base class defines the interface for resource management
    in MIMO-OFDM systems. Implementations can override the hooks to adapt
    configuration before system build (resource grid creation) and per batch
    during simulation.
    
    Theory:
        Resource management involves:
        
        1. Pre-build Configuration:
           - Set up resource grid structure
           - Configure pilot patterns
           - Initialize scheduling policies
           
        2. Runtime Directives:
           - User scheduling decisions
           - Power control adjustments
           - Link adaptation (MCS selection)
           - Interference coordination
           
        The resource manager can use feedback (e.g., BLER, SINR) to adapt
        resource allocation dynamically based on channel conditions and
        system performance.
    
    Subclasses should implement:
    - apply_pre_build(): Configure system before component construction
    - get_runtime_directives(): Return resource allocation decisions per batch
    """
    def apply_pre_build(self, config: dict) -> None:
        """
        Optional: mutate config prior to model component construction.
        
        This method is called before system components are built, allowing
        the resource manager to modify configuration parameters that affect
        resource grid structure, pilot patterns, etc.
        
        Theory:
            Pre-build configuration can include:
            - Pilot pattern selection
            - Resource grid dimensions
            - Pilot reuse configuration
            - Initial scheduling setup
            
        Args:
            config: System configuration dict to be modified.
        """
        return
    
    def get_runtime_directives(
        self,
        config: dict,
        ebno_db: float,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> ResourceDirectives:
        """
        Return per-batch runtime directives.
        
        This method is called for each batch during simulation to determine
        resource allocation decisions. The directives control user scheduling,
        power control, and other runtime parameters.
        
        Theory:
            Runtime directives can be based on:
            - Channel conditions (SNR, SINR)
            - User priorities and QoS requirements
            - Interference levels
            - Feedback from previous transmissions (BLER, throughput)
            - Link adaptation policies
            
            Adaptive resource management:
            - Good channels: Higher power, higher MCS
            - Poor channels: Lower power, lower MCS, or skip scheduling
            - Fairness: Balance resource allocation across users
            
        Args:
            config: System configuration dict (read-only, for reference).
            ebno_db: Current Eb/No in dB (can be used for link adaptation).
            feedback: Optional feedback dictionary containing:
                - BLER per UT
                - SINR per UT
                - Throughput per UT
                - Channel quality indicators
                Used for adaptive resource management.
                
        Returns:
            ResourceDirectives object containing resource allocation decisions
            for the current batch.
        """
        return ResourceDirectives()


class StaticResourceManager(ResourceManager):
    """
    Simple static resource manager that applies fixed scheduling and power.
    
    This implementation applies fixed resource allocation decisions that do
    not change during simulation. Useful for baseline comparisons and testing
    specific resource scenarios.
    """
    def __init__(
        self,
        active_ut_mask: Optional[List[int]] = None,
        per_ut_power: Optional[List[float]] = None,
        pilot_reuse_factor: Optional[int] = None,
        channel_model_type: Optional[str] = None,
    ):
        self._active_ut_mask = active_ut_mask
        self._per_ut_power = per_ut_power
        self._pilot_reuse_factor = pilot_reuse_factor
        self._channel_model_type = channel_model_type
    
    def apply_pre_build(self, config: dict) -> None:
        if self._channel_model_type is not None:
            config["channel_model_type"] = self._channel_model_type
        if self._pilot_reuse_factor is not None:
            config["pilot_reuse_factor"] = int(self._pilot_reuse_factor)
        if self._active_ut_mask is not None:
            config["active_ut_mask"] = list(self._active_ut_mask)
        if self._per_ut_power is not None:
            config["per_ut_power"] = list(self._per_ut_power)
    
    def get_runtime_directives(
        self,
        config: dict,
        ebno_db: float,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> ResourceDirectives:
        return ResourceDirectives(
            active_ut_mask=self._active_ut_mask,
            per_ut_power=self._per_ut_power,
            pilot_reuse_factor=self._pilot_reuse_factor,
        )


class RoundRobinResourceManager(ResourceManager):
    """
    Round-robin resource manager that cycles through users.
    
    Schedules one UT at a time (or a subset) in each batch, rotating through
    the available population to ensure fairness.
    """
    def __init__(self, num_active: int = 1):
        self.num_active = num_active
        self._current_index = 0
        
    def get_runtime_directives(
        self,
        config: dict,
        ebno_db: float,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> ResourceDirectives:
        num_ut = config.get("num_ut", 8)
        mask = [0] * num_ut
        
        for i in range(self.num_active):
            idx = (self._current_index + i) % num_ut
            mask[idx] = 1
            
        self._current_index = (self._current_index + self.num_active) % num_ut
        
        return ResourceDirectives(active_ut_mask=mask)


class MaxThroughputResourceManager(ResourceManager):
    """
    Maximum throughput resource manager.
    
    Schedules users with the best instantaneous channel conditions.
    Requires channel state information in the feedback.
    """
    def __init__(self, num_active: int = 1):
        self.num_active = num_active
        
    def get_runtime_directives(
        self,
        config: dict,
        ebno_db: float,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> ResourceDirectives:
        if not feedback or "h_hat" not in feedback:
            # Fallback to scheduling all if no CSI
            return ResourceDirectives(active_ut_mask=[1] * config.get("num_ut", 8))
            
        import tensorflow as tf
        h_hat = feedback["h_hat"] # [batch, num_rx, num_tx, num_streams, num_ofdm, fft_size]
        
        # Calculate channel energy per UT
        # Sum over Rx antennas, streams, time, freq
        power = tf.reduce_sum(tf.abs(h_hat)**2, axis=[1, 2, 4, 5, 6]) # [batch, num_tx]
        avg_power = tf.reduce_mean(power, axis=0) # [num_tx]
        
        # Select Top N users
        top_indices = tf.argsort(avg_power, direction='DESCENDING')[:self.num_active]
        top_indices = top_indices.numpy().tolist()
        
        mask = [0] * config.get("num_ut", 8)
        for idx in top_indices:
            mask[idx] = 1
            
        return ResourceDirectives(active_ut_mask=mask)


class ProportionalFairResourceManager(ResourceManager):
    """
    Proportional fair resource manager.
    
    Balances total throughput and fairness by scheduling users i that maximize
    R_i / T_i, where R_i is instantaneous rate and T_i is historical average rate.
    """
    def __init__(self, num_active: int = 1, alpha: float = 0.9):
        self.num_active = num_active
        self.alpha = alpha
        self.avg_rates = None
        
    def get_runtime_directives(
        self,
        config: dict,
        ebno_db: float,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> ResourceDirectives:
        num_ut = config.get("num_ut", 8)
        if self.avg_rates is None:
            self.avg_rates = np.ones(num_ut) * 1e-3 # Small epsilon
            
        if not feedback or "h_hat" not in feedback:
            return ResourceDirectives(active_ut_mask=[1] * num_ut)

        import tensorflow as tf
        h_hat = feedback["h_hat"]
        
        # Estimate instantaneous "rate" proportional to channel power
        # R_i ~ log2(1 + SNR_i)
        power = tf.reduce_sum(tf.abs(h_hat)**2, axis=[1, 2, 4, 5, 6])
        avg_power = tf.reduce_mean(power, axis=0).numpy()
        
        # Simple proxy for rate
        inst_rates = np.log2(1 + avg_power * (10**(ebno_db/10)))
        
        # PF metric: R_i / T_i
        pf_metric = inst_rates / self.avg_rates
        
        top_indices = np.argsort(pf_metric)[-self.num_active:]
        
        mask = [0] * num_ut
        for idx in top_indices:
            mask[idx] = 1
            
        # Update historical average rates
        # T_i = (1-alpha)*T_i + alpha*R_i (if scheduled)
        # T_i = (1-alpha)*T_i (if not scheduled)
        for i in range(num_ut):
            if mask[i] == 1:
                self.avg_rates[i] = (1 - self.alpha) * self.avg_rates[i] + self.alpha * inst_rates[i]
            else:
                self.avg_rates[i] = (1 - self.alpha) * self.avg_rates[i]
                
        return ResourceDirectives(active_ut_mask=mask)


