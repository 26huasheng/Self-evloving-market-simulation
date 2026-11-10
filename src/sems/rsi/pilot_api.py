"""Single-client guard for a pilot: only Decision calls, including retries."""

from sems.api_client import SEMSAPIClient


class PilotDecisionClient(SEMSAPIClient):
    LIMIT = 150

    def __init__(self, *, max_requests=LIMIT, **kwargs):
        if type(max_requests) is not int or not 0 <= max_requests <= self.LIMIT:
            raise ValueError("pilot request cap must be between 0 and 150")
        super().__init__(max_requests=max_requests, **kwargs)

    def chat(self, *, module, **kwargs):
        if module != "decision":
            raise ValueError("pilot permits Decision API calls only")
        return super().chat(module=module, **kwargs)
