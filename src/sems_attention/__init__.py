"""Neural Attention inference interface; training belongs to RSI."""

from .profiles.four_stage import ProfileExtractor, history_for_user, profile_input, validate_profile
from .profile_encoder import ProfileEncoder
from .environment_encoder import EnvironmentEncoder
from .lambda_layer import DynamicLambda
from .observation import Observation
from .actions import ToolAction, ActionSpace
from .policy import NeuralAttentionPolicy
from .trend_policy import TrendAttentionPolicy
from .runner import AttentionRunner

__all__ = [
    "ProfileExtractor",
    "history_for_user",
    "profile_input",
    "validate_profile",
    "ProfileEncoder",
    "EnvironmentEncoder",
    "DynamicLambda",
    "Observation",
    "ToolAction",
    "ActionSpace",
    "NeuralAttentionPolicy",
    "TrendAttentionPolicy",
    "AttentionRunner",
]
