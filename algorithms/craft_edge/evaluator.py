# CRAFT-Edge Recovery Outcome Evaluator.
#
# Quantitatively measures recovery benefit, operational cost, and calculates
# scalar rewards for online policy updating:
#     reward = w_benefit * benefit - w_cost * cost
#
# Benefit is computed as normalized relative improvement in target health metrics
# (CPU, Memory, Latency, Packet Loss, Node Liveness). Cost reflects operational penalty.

from dataclasses import dataclass, field
import logging
from typing import Dict, Any, Optional

from .policy import ACTION_OPERATIONAL_COSTS, RecoveryAction

logger = logging.getLogger("CRAFTEdge.Evaluator")


@dataclass
class RecoveryOutcome:
    # Structured outcome evaluation result following action execution.
    action: str
    root_cause_category: str
    target_metric: str
    metric_before: float
    metric_after: float
    absolute_improvement: float
    relative_improvement: float
    benefit: float
    cost: float
    reward: float
    success: bool
    execution_applied: bool = True
    details: Dict[str, Any] = field(default_factory=dict)



class RecoveryOutcomeEvaluator:
    # Quantitative outcome evaluation engine assessing recovery actions.
    # Operates across simulation states and shadow models with zero dataset mutation.

    def __init__(
        self,
        benefit_weight: float = 1.0,
        cost_weight: float = 0.3,
        success_threshold_ratio: float = 0.08,  # Minimum 8% relative improvement for success
    ):
        self.benefit_weight = float(benefit_weight)
        self.cost_weight = float(cost_weight)
        self.success_threshold_ratio = float(success_threshold_ratio)

    def determine_target_metric(
        self,
        root_cause_category: str,
        action: str,
        pre_metrics: Dict[str, float],
    ) -> str:
        # Maps root-cause and executed action to the primary physical target metric.
        cause = root_cause_category.upper()
        act = action.upper()

        if cause == "NODE_FAILURE" or act == RecoveryAction.EDGE_NODE_FAILOVER.value:
            return "latency"
        elif cause == "CPU_OVERLOAD" or act == RecoveryAction.WORKLOAD_REDISTRIBUTION.value:
            return "cpu_utilization"
        elif cause == "MEMORY_OVERLOAD" or act == RecoveryAction.TASK_MIGRATION.value:
            return "memory_utilization"
        elif cause in ["NETWORK_CONGESTION", "TRANSPORT_FLOOD"] or act == RecoveryAction.TRAFFIC_REROUTING.value:
            if pre_metrics.get("packet_loss", 0.0) > 5.0:
                return "packet_loss"
            return "latency"
        elif cause == "HIGH_LATENCY":
            return "latency"
        else:
            # Pick metric with highest relative elevation in pre_metrics
            cpu = pre_metrics.get("cpu_utilization", 0.0)
            mem = pre_metrics.get("memory_utilization", 0.0)
            lat = pre_metrics.get("latency", 0.0)
            if cpu >= 70.0:
                return "cpu_utilization"
            if mem >= 70.0:
                return "memory_utilization"
            if lat >= 50.0:
                return "latency"
            return "cpu_utilization"

    def evaluate(
        self,
        action: str,
        root_cause_category: str,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
        is_node_failed_before: bool = False,
        is_node_failed_after: bool = False,
        action_cost_override: Optional[float] = None,
        execution_applied: bool = True,
    ) -> RecoveryOutcome:
        # Evaluates the quantitative recovery outcome given before/after system metrics.
        # Resolves target metric
        target_metric = self.determine_target_metric(root_cause_category, action, pre_metrics)

        val_before = float(pre_metrics.get(target_metric, 0.0))
        val_after = float(post_metrics.get(target_metric, 0.0))

        # Operational recovery cost in [0.0, 1.0]
        if action_cost_override is not None:
            cost = float(action_cost_override)
        else:
            cost = float(ACTION_OPERATIONAL_COSTS.get(action, 0.25))

        # Explicit handling for unexecutable / rejected actions
        if not execution_applied:
            benefit = -0.40  # Negative failure penalty
            raw_reward = (self.benefit_weight * benefit) - (self.cost_weight * cost)
            reward = max(-1.0, min(1.0, raw_reward))
            outcome = RecoveryOutcome(
                action=action,
                root_cause_category=root_cause_category,
                target_metric=target_metric,
                metric_before=round(val_before, 2),
                metric_after=round(val_after, 2),
                absolute_improvement=0.0,
                relative_improvement=0.0,
                benefit=round(benefit, 4),
                cost=round(cost, 4),
                reward=round(reward, 4),
                success=False,
                execution_applied=False,
                details={
                    "benefit_weight": self.benefit_weight,
                    "cost_weight": self.cost_weight,
                    "execution_applied": False,
                    "failure_reason": "ACTION_EXECUTION_REJECTED_OR_FAILED",
                    "pre_metrics": pre_metrics,
                    "post_metrics": post_metrics,
                }
            )
            logger.warning(
                "[Outcome Evaluation] Action '%s' execution rejected/failed. Benefit=%.2f, Cost=%.2f, Reward=%.2f, Success=False",
                action, benefit, cost, reward
            )
            return outcome

        # Absolute improvement: for degradation metrics (CPU, Memory, Latency, Loss),
        # lower is better, so improvement = before - after
        abs_imp = val_before - val_after
        rel_imp = abs_imp / max(1e-4, val_before)

        # 1. Recovery Benefit formulation in [-1.0, 1.0]
        if is_node_failed_before and not is_node_failed_after:
            # Full recovery of crashed/failed node
            benefit = 1.0
        elif is_node_failed_after:
            # Node remains crashed / failed
            benefit = -1.0
        elif action == RecoveryAction.NO_ACTION.value:
            # Passive action: neutral benefit if degradation mild, penalty if severe
            if val_before > 75.0:
                benefit = -0.3  # Inaction penalty when system was degraded
            else:
                benefit = 0.0
        else:
            # Proportional improvement clamped between -1.0 and 1.0
            benefit = max(-1.0, min(1.0, rel_imp))

        # 2. Scalar reward formulation
        raw_reward = (self.benefit_weight * benefit) - (self.cost_weight * cost)
        reward = max(-1.0, min(1.0, raw_reward))

        # 3. Success condition
        is_success = bool(benefit >= self.success_threshold_ratio and reward > -0.05)
        if action == RecoveryAction.NO_ACTION.value and val_before < 50.0:
            is_success = True

        outcome = RecoveryOutcome(
            action=action,
            root_cause_category=root_cause_category,
            target_metric=target_metric,
            metric_before=round(val_before, 2),
            metric_after=round(val_after, 2),
            absolute_improvement=round(abs_imp, 2),
            relative_improvement=round(rel_imp * 100.0, 2),
            benefit=round(benefit, 4),
            cost=round(cost, 4),
            reward=round(reward, 4),
            success=is_success,
            execution_applied=True,
            details={
                "benefit_weight": self.benefit_weight,
                "cost_weight": self.cost_weight,
                "execution_applied": True,
                "pre_metrics": pre_metrics,
                "post_metrics": post_metrics,
            }
        )

        logger.debug(
            "[Outcome Evaluation] Action '%s' for '%s': Target=%s, Before=%.1f -> After=%.1f, "
            "Benefit=%.3f, Cost=%.3f, Reward=%.3f, Success=%s",
            action, root_cause_category, target_metric, val_before, val_after,
            benefit, cost, reward, is_success
        )
        return outcome
