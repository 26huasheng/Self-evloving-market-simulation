"""Single public tool contract and Attention action enumeration."""

from itertools import product
from copy import deepcopy

TOOLS = {
    "get_product_details": {
        "kind": "read",
        "params": {"asin": "shelf_asin", "section": ["overview", "metadata"]},
    },
    "get_price_info": {
        "kind": "read",
        "params": {"asin": "shelf_asin", "view": ["offers", "history"], "window_days": [30, 90]},
    },
    "get_rating_info": {"kind": "read", "params": {"asin": "shelf_asin"}},
    "get_reviews": {
        "kind": "read",
        "params": {
            "asin": "shelf_asin",
            "sort": ["recent", "oldest"],
            "rating_filter": ["all", "positive", "negative", "1", "2", "3", "4", "5"],
        },
    },
    "get_rank_info": {"kind": "read", "params": {"asin": "shelf_asin"}},
    "sort_shelf": {
        "kind": "shelf",
        "params": {
            "sort_by": ["default", "price", "rating", "rating_count"],
            "order": ["asc", "desc"],
        },
    },
    "purchase_product": {"kind": "write", "params": {"asin": "shelf_asin"}},
    "rate_product": {"kind": "write", "params": {"asin": "shelf_asin", "rating": [1, 2, 3, 4, 5]}},
    "publish_review": {"kind": "write", "params": {"asin": "shelf_asin", "text": "nonempty_text"}},
}


class ToolRegistry:
    def __init__(self, session):
        self.session = session

    def schema(self):
        s = deepcopy(TOOLS)
        if not self.session.market.rank_enabled:
            s.pop("get_rank_info")
        return s

    def product_index(self):
        return {i: a for i, a in enumerate(self.session.market.data.asins)}

    def asin(self, index):
        return self.product_index()[index]

    def actions(self):
        out = [{"tool": "STOP", "args": {}, "cost": 0, "allowed": True}]
        for name, spec in self.schema().items():
            if spec["kind"] == "write":
                continue
            choices = {
                k: (self.session.market.data.asins if v == "shelf_asin" else v)
                for k, v in spec["params"].items()
            }
            for vals in product(*choices.values()):
                args = dict(zip(choices, vals))
                if (
                    name == "get_price_info"
                    and args["view"] == "offers"
                    and args["window_days"] != 30
                ):
                    continue
                if name == "get_price_info" and args["view"] == "offers":
                    args.pop("window_days")
                if name == "sort_shelf" and args["sort_by"] == "default" and args["order"] != "asc":
                    continue
                allowed = (
                    self.session.can_sort(args["sort_by"], args["order"])
                    if name == "sort_shelf"
                    else self.session.remaining > 0
                )
                if (
                    name == "get_reviews"
                    and self.session.pages.get((args["asin"], args["sort"], args["rating_filter"]))
                    == -1
                ):
                    allowed = False
                out.append(
                    {
                        "tool": name,
                        "args": args,
                        "cost": 0 if spec["kind"] == "shelf" else 1,
                        "allowed": allowed,
                    }
                )
        return out

    def call(self, name, **args):
        return self.session.call(name, **args)
