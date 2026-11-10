"""SEMS v2 modules; load optional inference dependencies only when requested."""

from importlib import import_module

_EXPORTS = {
    "DecisionModel": (".decision", "DecisionModel"),
    "OpenAICompatibleClient": (".decision", "OpenAICompatibleClient"),
    "SEMSAPIClient": (".api_client", "SEMSAPIClient"),
    "SimulationRunner": (".simulation_runner", "SimulationRunner"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
