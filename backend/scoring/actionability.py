from typing import Tuple
from backend.interfaces import EventPriority, CameraStatus, Actionability

class ActionabilityMatrix:
    """
    Computes composite Actionability rating independently from Event Priority and Camera Reliability.
    Defines recommended operational handling based on observation quality and event significance.
    """

    @staticmethod
    def evaluate(priority: EventPriority, reliability_status: CameraStatus) -> Tuple[Actionability, str]:
        """
        Returns Tuple[Actionability (HIGH|MEDIUM|LOW), action_recommendation_text].
        """
        # OFFLINE Camera handling
        if reliability_status == CameraStatus.OFFLINE:
            return (
                Actionability.LOW,
                "Camera offline or missing signal. Recommend immediate technician dispatch / camera verification."
            )

        # CRITICAL Priority Handling
        if priority == EventPriority.CRITICAL:
            if reliability_status in [CameraStatus.GOOD, CameraStatus.DEGRADED]:
                return (
                    Actionability.HIGH,
                    "Immediate operator verification and response required. Event priority is CRITICAL."
                )
            else:  # POOR
                return (
                    Actionability.MEDIUM,
                    "Critical priority event observed on POOR quality feed. Verify camera condition before escalation."
                )

        # HIGH Priority Handling
        elif priority == EventPriority.HIGH:
            if reliability_status == CameraStatus.GOOD:
                return (
                    Actionability.HIGH,
                    "High priority event confirmed on clear camera feed. Operator verification required."
                )
            elif reliability_status == CameraStatus.DEGRADED:
                return (
                    Actionability.MEDIUM,
                    "High priority event observed on DEGRADED camera feed. Verify camera condition before escalation."
                )
            else:  # POOR
                return (
                    Actionability.MEDIUM,
                    "High priority event on POOR quality feed. Secondary verification recommended."
                )

        # MEDIUM Priority Handling
        elif priority == EventPriority.MEDIUM:
            if reliability_status in [CameraStatus.GOOD, CameraStatus.DEGRADED]:
                return (
                    Actionability.MEDIUM,
                    "Medium operational priority event. Standard operator review queued."
                )
            else:
                return (
                    Actionability.LOW,
                    "Medium priority event on low quality camera feed. Monitor post."
                )

        # LOW Priority Handling
        else:  # LOW
            return (
                Actionability.LOW,
                "Low priority routine observation. Logged for standard audit."
            )
