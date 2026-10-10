# CRAFT-Edge Adaptive Efficacy Recovery Policy.
#
# Implements an interpretable online Q-value action-efficacy policy:
#     Q_new(cause, action) = Q_old(cause, action) + alpha * (reward - Q_old(cause, action))
#
# Manages:
# 1. Standard recovery actions matching physical edge simulation capabilities.
# 2. Configurable operational recovery costs.
# 3. Optimistic / prior initialization for unseen state-action pairs.
# 4. Controlled exploration via epsilon-greedy selection with decay.
# 5. Online Q-table updates driven by measured outcome reward.

from enum import Enum
import random
import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger("CRAFTEdge.Policy")


class RecoveryAction(str, Enum):
    # Explicit set of recovery actions supported by edge simulation and state models.
    NO_ACTION = "NO_ACTION"
    RESOURCE_REBALANCING = "RESOURCE_REBALANCING"
    WORKLOAD_REDISTRIBUTION = "WORKLOAD_REDISTRIBUTION"
    TRAFFIC_REROUTING = "TRAFFIC_REROUTING"
    TASK_MIGRATION = "TASK_MIGRATION"
    EDGE_NODE_FAILOVER = "EDGE_NODE_FAILOVER"

    @classmethod
    def all_actions(cls) -> List[str]:
        return [a.value for a in cls]


# Operational execution and migration costs associated with each recovery action (normalized in [0.0, 1.0]).
ACTION_OPERATIONAL_COSTS: Dict[str, float] = {
    RecoveryAction.NO_ACTION.value: 0.0,
    RecoveryAction.RESOURCE_REBALANCING.value: 0.15,
    RecoveryAction.WORKLOAD_REDISTRIBUTION.value: 0.20,
    RecoveryAction.TRAFFIC_REROUTING.value: 0.25,
    RecoveryAction.TASK_MIGRATION.value: 0.50,
    RecoveryAction.EDGE_NODE_FAILOVER.value: 0.80,
}


def classify_root_cause(root_cause_metric: str) -> str:
    # Classifies a granular telemetry feature or fault label into a canonical
    # root-cause category for policy indexing.
    cause_clean = str(root_cause_metric).lower().strip()

    if "cpu" in cause_clean:
        return "CPU_OVERLOAD"
    elif "mem" in cause_clean:
        return "MEMORY_OVERLOAD"
    elif "net" in cause_clean or "loss" in cause_clean:
        return "NETWORK_CONGESTION"
    elif "lat" in cause_clean:
        return "HIGH_LATENCY"
    elif any(proto in cause_clean for proto in ["tcp", "udp", "icmp", "arp"]):
        return "TRANSPORT_FLOOD"
    elif any(proto in cause_clean for proto in ["http", "mqtt", "dns", "mbtcp"]):
        return "APPLICATION_EXPLOIT"
    elif "fail" in cause_clean or "crash" in cause_clean:
        return "NODE_FAILURE"
    else:
        return "GENERIC_RESOURCE_FAULT"


class AdaptiveEfficacyPolicy:
    # Online adaptive action-efficacy policy using Q-learning updates and configurable exploration.
    # Maintains bounded state, provides fully interpretable tabular updates, and prevents
    # rapid consecutive anomaly bursts from prematurely collapsing exploration.

    def __init__(
        self,
        learning_rate: float = 0.2,
        epsilon: float = 0.15,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        decay_interval: int = 1,
        decay_mode: str = "interval",
        default_q: float = 0.50,
        action_costs: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        self.alpha = float(learning_rate)
        self.initial_epsilon = float(epsilon)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.decay_interval = max(1, int(decay_interval))
        self.decay_mode = str(decay_mode).lower().strip()
        if self.decay_mode not in ("interval", "episode"):
            self.decay_mode = "interval"

        self.default_q = float(default_q)
        self.action_costs = action_costs or dict(ACTION_OPERATIONAL_COSTS)
        self.rng = random.Random(seed)

        self.update_count: int = 0
        self.episode_count: int = 0

        # Tabular efficacy storage: (cause, action) -> Q_value
        self.q_table: Dict[Tuple[str, str], float] = {}
        # Execution statistics: (cause, action) -> {"attempts": int, "successes": int, "last_reward": float}
        self.action_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Domain initialization priors: slight sensible optimism for logical matches
        self.domain_priors: Dict[Tuple[str, str], float] = {
            ("CPU_OVERLOAD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value): 0.65,
            ("CPU_OVERLOAD", RecoveryAction.RESOURCE_REBALANCING.value): 0.55,
            ("MEMORY_OVERLOAD", RecoveryAction.TASK_MIGRATION.value): 0.65,
            ("NETWORK_CONGESTION", RecoveryAction.TRAFFIC_REROUTING.value): 0.65,
            ("TRANSPORT_FLOOD", RecoveryAction.TRAFFIC_REROUTING.value): 0.60,
            ("APPLICATION_EXPLOIT", RecoveryAction.RESOURCE_REBALANCING.value): 0.55,
            ("NODE_FAILURE", RecoveryAction.EDGE_NODE_FAILOVER.value): 0.70,
        }

    def decay_epsilon(self) -> float:
        # Applies one step of exponential epsilon decay bounded by epsilon_min.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def step_episode(self) -> float:
        # Records completion of a fault episode and decays epsilon if decay_mode is 'episode'.
        self.episode_count += 1
        if self.decay_mode == "episode" and (self.episode_count % self.decay_interval == 0):
            self.decay_epsilon()
        return self.epsilon

    def get_q_value(self, cause: str, action: str) -> float:
        # Retrieves current Q(cause, action), initializing if unseen.
        key = (cause, action)
        if key not in self.q_table:
            # Unseen action initialization: domain prior if defined, else default optimistic Q0
            init_val = self.domain_priors.get(key, self.default_q)
            self.q_table[key] = float(init_val)
            self.action_stats[key] = {"attempts": 0, "successes": 0, "last_reward": 0.0}
        return self.q_table[key]

    def set_q_value(self, cause: str, action: str, value: float) -> None:
        key = (cause, action)
        self.q_table[key] = float(value)
        if key not in self.action_stats:
            self.action_stats[key] = {"attempts": 0, "successes": 0, "last_reward": 0.0}

    def select_action(
        self,
        cause: str,
        explore: bool = True,
        candidate_actions: Optional[List[str]] = None,
    ) -> Tuple[str, bool]:
        # Selects a recovery action for the diagnosed root cause.
        # Uses epsilon-greedy exploration.
        # Returns (selected_action, is_explored).
        actions = candidate_actions or RecoveryAction.all_actions()
        if not actions:
            return RecoveryAction.NO_ACTION.value, False

        # Ensure all candidate Q-values are initialized
        for act in actions:
            _ = self.get_q_value(cause, act)

        is_explored = False
        if explore and (self.rng.random() < self.epsilon):
            # Exploration: uniformly select among candidate actions
            selected_action = self.rng.choice(actions)
            is_explored = True
            logger.debug(
                "[Policy Explore] Cause '%s': explored action '%s' (eps=%.3f)",
                cause, selected_action, self.epsilon
            )
        else:
            # Exploitation: select action with maximum Q-value
            best_q = -float("inf")
            best_actions = []
            for act in actions:
                q_val = self.get_q_value(cause, act)
                if q_val > best_q:
                    best_q = q_val
                    best_actions = [act]
                elif abs(q_val - best_q) < 1e-6:
                    best_actions.append(act)

            # Deterministic tie-breaking using RNG
            selected_action = self.rng.choice(best_actions)
            logger.debug(
                "[Policy Exploit] Cause '%s': selected best action '%s' (Q=%.3f)",
                cause, selected_action, best_q
            )

        return selected_action, is_explored

    def update(
        self,
        cause: str,
        action: str,
        reward: float,
        success: bool = True,
        is_episode_end: bool = False,
    ) -> float:
        # Executes online Q-value efficacy update:
        #     Q_new = Q_old + alpha * (reward - Q_old)
        # Clamps Q_value to [-1.0, 1.0].
        # Updates exploration rate via configurable schedule.
        q_old = self.get_q_value(cause, action)
        reward_clamped = max(-1.0, min(1.0, float(reward)))

        # Bellman / TD-style efficacy update on immediate outcome reward
        q_new = q_old + self.alpha * (reward_clamped - q_old)
        q_new = max(-1.0, min(1.0, q_new))
        self.set_q_value(cause, action, q_new)

        # Update action statistics
        key = (cause, action)
        self.action_stats[key]["attempts"] += 1
        if success:
            self.action_stats[key]["successes"] += 1
        self.action_stats[key]["last_reward"] = round(reward_clamped, 4)

        # Apply configurable epsilon decay
        self.update_count += 1
        if self.decay_mode == "interval":
            if self.update_count % self.decay_interval == 0:
                self.decay_epsilon()
        elif self.decay_mode == "episode":
            if is_episode_end:
                self.step_episode()

        logger.info(
            "[Policy Update] (%s, %s): Q_old=%.3f -> Q_new=%.3f (reward=%.3f, eps=%.3f, updates=%d, episodes=%d)",
            cause, action, q_old, q_new, reward_clamped, self.epsilon, self.update_count, self.episode_count
        )
        return q_new

    def get_action_cost(self, action: str) -> float:
        return self.action_costs.get(action, 0.25)

    def get_stats(self) -> Dict[str, Any]:
        table_dump = {f"{c}::{a}": round(q, 4) for (c, a), q in self.q_table.items()}
        return {
            "q_table": table_dump,
            "epsilon": round(self.epsilon, 4),
            "learning_rate": self.alpha,
            "decay_interval": self.decay_interval,
            "decay_mode": self.decay_mode,
            "update_count": self.update_count,
            "episode_count": self.episode_count,
            "total_updates": sum(st["attempts"] for st in self.action_stats.values()),
            "action_stats": {f"{c}::{a}": st for (c, a), st in self.action_stats.items()},
        }

    def reset(self) -> None:
        self.q_table.clear()
        self.action_stats.clear()
        self.update_count = 0
        self.episode_count = 0
        self.epsilon = self.initial_epsilon
