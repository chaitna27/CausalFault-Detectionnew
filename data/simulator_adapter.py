# Simulator Adapter for Streaming Telemetry.
#
# Adapts the discrete-event SimPy EdgeSimulation environment to the unified
# DataSource interface, generating TelemetryRecord streams for real-time detection,
# causal analysis, and adaptive recovery experiments.

from typing import Generator, Optional, TYPE_CHECKING
import logging
from .telemetry import TelemetryRecord, DataSource

if TYPE_CHECKING:
    from simulation.edge_simulation import EdgeSimulation

logger = logging.getLogger(__name__)


class SimulatorAdapter(DataSource):
    # Adapter converting SimPy EdgeSimulation steps into a streaming TelemetryRecord feed.

    def __init__(
        self,
        simulation: Optional["EdgeSimulation"] = None,
        max_steps: int = 300,
        seed: int = 42,
    ):
        self.max_steps = max_steps
        self.seed = seed
        if simulation is not None:
            self.simulation = simulation
        else:
            from simulation.edge_simulation import EdgeSimulation
            self.simulation = EdgeSimulation(seed=seed)

    def get_dataset_name(self) -> str:
        return "SimPy_Edge_IoT_Simulation"

    def stream_telemetry(self) -> Generator[TelemetryRecord, None, None]:
        # Yield telemetry records step-by-step from all edge nodes in the simulation.
        logger.info(f"Starting simulation telemetry stream (max_steps={self.max_steps}, seed={self.seed})...")

        for _ in range(self.max_steps):
            records = self.simulation.step()
            for rec in records:
                yield rec

