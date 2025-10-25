"""
Thermal Management Utility for Apple Silicon GPU workloads.

Provides intelligent thermal throttling detection and prevention
for long-running image generation tasks.
"""

import time
import statistics
from typing import List, Optional, Dict
from dataclasses import dataclass
from collections import deque


@dataclass
class ThermalState:
    """Current thermal state of the system."""
    is_throttled: bool
    performance_ratio: float  # 1.0 = normal, < 1.0 = throttled
    recent_times: List[float]
    baseline_time: float


class ThermalManager:
    """
    Manages thermal throttling detection and prevention for GPU workloads.

    Monitors generation timing to detect thermal throttling in real-time,
    then automatically inserts cooling breaks to prevent critical throttling.

    Example:
        thermal_mgr = ThermalManager(baseline_time_per_step=5.0)

        for image in images:
            # Check if we need a cooling break
            if thermal_mgr.should_cool():
                print("Taking cooling break...")
                thermal_mgr.cooling_break()

            # Generate image and record timing
            start = time.time()
            generate_image()
            elapsed = time.time() - start

            thermal_mgr.record_timing(elapsed)
    """

    def __init__(
        self,
        baseline_time_per_step: Optional[float] = None,
        throttle_threshold: float = 1.5,
        critical_threshold: float = 3.0,
        window_size: int = 5,
        cooling_duration: int = 30,
    ):
        """
        Initialize thermal manager.

        Args:
            baseline_time_per_step: Expected normal performance (seconds/step).
                                   If None, will be learned from first few samples.
            throttle_threshold: Slowdown factor to trigger cooling (1.5 = 50% slower)
            critical_threshold: Slowdown factor indicating critical throttling
            window_size: Number of recent timings to track
            cooling_duration: How long to pause during cooling breaks (seconds)
        """
        self.baseline = baseline_time_per_step
        self.throttle_threshold = throttle_threshold
        self.critical_threshold = critical_threshold
        self.cooling_duration = cooling_duration

        self.timings = deque(maxlen=window_size)
        self.all_timings: List[float] = []
        self.cooling_breaks_taken = 0
        self.total_cooling_time = 0.0

    def record_timing(self, elapsed_time: float) -> None:
        """Record timing from a generation step."""
        self.timings.append(elapsed_time)
        self.all_timings.append(elapsed_time)

        # Auto-learn baseline from first few samples if not provided
        if self.baseline is None and len(self.all_timings) >= 3:
            self.baseline = statistics.median(list(self.timings))

    def get_thermal_state(self) -> Optional[ThermalState]:
        """Get current thermal state."""
        if not self.timings or self.baseline is None:
            return None

        recent_avg = statistics.mean(self.timings)
        performance_ratio = self.baseline / recent_avg
        is_throttled = performance_ratio < (1.0 / self.throttle_threshold)

        return ThermalState(
            is_throttled=is_throttled,
            performance_ratio=performance_ratio,
            recent_times=list(self.timings),
            baseline_time=self.baseline,
        )

    def should_cool(self) -> bool:
        """
        Check if we should take a cooling break.

        Returns True if performance has degraded significantly,
        indicating thermal throttling is occurring.
        """
        state = self.get_thermal_state()
        if state is None:
            return False

        # If recent performance is significantly worse than baseline, cool down
        slowdown_factor = max(state.recent_times) / state.baseline_time
        return slowdown_factor >= self.throttle_threshold

    def cooling_break(self, duration: Optional[int] = None) -> None:
        """
        Take a cooling break to let GPU temperature drop.

        Args:
            duration: How long to pause (seconds). Uses default if not provided.
        """
        pause_time = duration if duration is not None else self.cooling_duration

        print(f"  🌡️  Thermal throttling detected - taking {pause_time}s cooling break...")
        time.sleep(pause_time)

        self.cooling_breaks_taken += 1
        self.total_cooling_time += pause_time

        print(f"  ✓ Cooling complete (break #{self.cooling_breaks_taken})")

    def get_stats(self) -> Dict:
        """Get thermal management statistics."""
        if not self.all_timings:
            return {}

        state = self.get_thermal_state()

        return {
            "total_generations": len(self.all_timings),
            "cooling_breaks_taken": self.cooling_breaks_taken,
            "total_cooling_time_seconds": self.total_cooling_time,
            "baseline_time_per_step": self.baseline,
            "current_performance_ratio": state.performance_ratio if state else None,
            "is_currently_throttled": state.is_throttled if state else False,
            "avg_time_per_step": statistics.mean(self.all_timings),
            "min_time_per_step": min(self.all_timings),
            "max_time_per_step": max(self.all_timings),
        }

    def print_stats(self) -> None:
        """Print thermal management statistics."""
        stats = self.get_stats()
        if not stats:
            print("No thermal statistics available yet.")
            return

        print("\n" + "=" * 70)
        print("🌡️  THERMAL MANAGEMENT STATISTICS")
        print("=" * 70)
        print(f"Total Generations:        {stats['total_generations']}")
        print(f"Cooling Breaks Taken:     {stats['cooling_breaks_taken']}")
        print(f"Total Cooling Time:       {stats['total_cooling_time_seconds']:.1f}s ({stats['total_cooling_time_seconds']/60:.1f} min)")
        print(f"\nPerformance:")
        print(f"  Baseline:               {stats['baseline_time_per_step']:.2f}s/step")
        print(f"  Current Ratio:          {stats['current_performance_ratio']:.2f}x" if stats['current_performance_ratio'] else "  Current Ratio:          N/A")
        print(f"  Currently Throttled:    {'Yes' if stats['is_currently_throttled'] else 'No'}")
        print(f"\nTiming Range:")
        print(f"  Average:                {stats['avg_time_per_step']:.2f}s/step")
        print(f"  Min:                    {stats['min_time_per_step']:.2f}s/step")
        print(f"  Max:                    {stats['max_time_per_step']:.2f}s/step")
        print("=" * 70 + "\n")


class BatchThermalManager(ThermalManager):
    """
    Thermal manager optimized for batch processing.

    Automatically inserts cooling breaks between batches
    to maintain optimal thermal performance.
    """

    def __init__(
        self,
        batch_size: int = 10,
        batch_cooling_duration: int = 60,
        **kwargs
    ):
        """
        Initialize batch thermal manager.

        Args:
            batch_size: Number of generations per batch before cooling
            batch_cooling_duration: Cooling time between batches (seconds)
            **kwargs: Passed to ThermalManager
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.batch_cooling_duration = batch_cooling_duration
        self.current_batch_count = 0

    def record_timing(self, elapsed_time: float) -> None:
        """Record timing and track batch count."""
        super().record_timing(elapsed_time)
        self.current_batch_count += 1

    def should_cool(self) -> bool:
        """Check if cooling is needed (batch-based or throttle-based)."""
        # Cool at batch boundaries
        if self.current_batch_count >= self.batch_size:
            return True

        # Or if throttling detected
        return super().should_cool()

    def cooling_break(self, duration: Optional[int] = None) -> None:
        """Take cooling break and reset batch counter."""
        # Use batch cooling duration if at batch boundary
        if self.current_batch_count >= self.batch_size and duration is None:
            duration = self.batch_cooling_duration

        super().cooling_break(duration)
        self.current_batch_count = 0
