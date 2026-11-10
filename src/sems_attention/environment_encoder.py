import torch
from torch import nn


class EnvironmentEncoder(nn.Module):
    def __init__(self, structure=None):
        super().__init__()
        self.structure = structure or {}
        self.product = nn.Sequential(nn.Linear(1168, 128), nn.GELU(), nn.Linear(128, 64))
        self.shelf = nn.Sequential(
            nn.Linear(136, 128), nn.GELU(), nn.Linear(128, 64), nn.LayerNorm(64)
        )
        if self.structure.get("query_context_v1"):
            self.condition_adapter = nn.Linear(4, 64, bias=False)
            nn.init.zeros_(self.condition_adapter.weight)

    def forward(self, observation, embeddings):
        rows = []
        for asin in observation.order:
            title, detail, reviews = observation.product_texts(asin)
            rows.append(
                torch.cat(
                    [
                        torch.as_tensor(embeddings.encode(title)),
                        torch.as_tensor(embeddings.encode(detail)) if detail else torch.zeros(384),
                        torch.as_tensor(embeddings.mean(reviews)),
                        torch.as_tensor(observation.product_features(asin)),
                    ]
                )
            )
        raw = torch.stack(rows)
        h = self.product(raw)
        e = self.shelf(
            torch.cat([h.mean(0), h.max(0).values, torch.as_tensor(observation.shelf_stats())])
        )
        if self.structure.get("query_context_v1"):
            n = max(len(observation.order), 1)
            queried = {
                r["action"].get("asin") for r in observation.results if r["action"].get("asin")
            }
            seen_reviews = sum(bool(observation.reviews.get(a)) for a in observation.order)
            free_sorts = sum(
                r["action"].get("tool_name") == "sort_shelf" for r in observation.results
            )
            context = e.new_tensor(
                [
                    1 - seen_reviews / n,
                    len(queried) / n,
                    min(len(observation.results) / max(observation.initial_budget, 1), 1),
                    min(free_sorts / 3, 1),
                ]
            )
            e = e + self.condition_adapter(context)
        return h, e, raw
