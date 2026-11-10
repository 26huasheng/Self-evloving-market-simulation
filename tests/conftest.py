"""Generate a small synthetic Case outside the source tree for integration tests."""

import json
import os
from pathlib import Path
import tempfile
import zipfile

import torch

_fixture_directory = None


def pytest_configure(config):
    global _fixture_directory
    torch.set_num_threads(1)
    if os.getenv("SEMS_CASE_PATH"):
        return
    _fixture_directory = tempfile.TemporaryDirectory(prefix="sems-tests-")
    archive = Path(_fixture_directory.name) / "synthetic_case.zip"
    t0 = 1600000000000
    day = 86400000
    users = ["user_a", "user_b", "user_c"]
    files = {
        "case/case.json": json.dumps(
            {
                "case_id": "synthetic_test_case",
                "t0_timestamp_ms": t0,
                "review_window_end_timestamp_ms": t0 + 181 * day,
            }
        )
    }
    shelf = []
    histories = {user: [] for user in users}
    for index in range(4):
        asin = f"TEST{index}"
        shelf.append(
            {
                "product_id": asin,
                "title": f"Test product {index}",
                "product_record_path": f"{asin}/product.json",
                "smoothed_history_path": f"{asin}/series.json",
                "product_reviews_path": f"{asin}/reviews.jsonl",
            }
        )
        static = {
            "description": f"Synthetic description {index}.",
            "features": ["Test feature"],
            "brand": "Test brand",
            "model": str(index),
            "size": "standard",
            "color": "black",
            "manufacturer": "Test maker",
        }
        files[f"case/{asin}/product.json"] = json.dumps({"keepa": {"static": static}})
        prices = [
            {"timestamp_ms": t0 + n * day, "value": 1000 + index * 500 + n, "status": "observed"}
            for n in [-80, -20, *range(181)]
        ]
        series = {
            "csv:NEW": prices,
            "csv:USED": [],
            "csv:LISTPRICE": [],
            "csv:SALES": [{"timestamp_ms": t0, "value": 100 + index, "status": "observed"}],
            "salesRanks:SALES_RANK_CATEGORY:1": [
                {"timestamp_ms": t0, "value": 100 + index, "status": "observed"}
            ],
            "csv:RATING": [{"timestamp_ms": t0, "value": 4.2, "status": "observed"}],
            "csv:COUNT_REVIEWS": [{"timestamp_ms": t0, "value": 16, "status": "observed"}],
        }
        files[f"case/{asin}/series.json"] = json.dumps({"series": series})
        reviews = []
        for n in range(16):
            user = users[n % 3]
            row = {
                "asin": asin,
                "review_id": f"{asin}-past-{n}",
                "user_id": user,
                "timestamp_ms": t0 - (n + 1) * day,
                "rating": n % 5 + 1,
                "text": f"Synthetic historical review {n} about {asin}.",
            }
            reviews.append(row)
            histories[user].append(row)
        if index < 3:
            for n, offset in enumerate((0, 5, 170)):
                reviews.append(
                    {
                        "asin": asin,
                        "review_id": f"{asin}-future-{n}",
                        "user_id": users[index],
                        "timestamp_ms": t0 + offset * day + index * 1000,
                        "rating": 4,
                        "text": "Synthetic held-out review.",
                    }
                )
        files[f"case/{asin}/reviews.jsonl"] = "\n".join(json.dumps(row) for row in reviews)
    files["case/shelf.jsonl"] = "\n".join(json.dumps(row) for row in shelf)
    for user, rows in histories.items():
        files[f"case/user_profiles/{user}.jsonl"] = "\n".join(json.dumps(row) for row in rows)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for name, content in files.items():
            output.writestr(name, content)
    os.environ["SEMS_CASE_PATH"] = str(archive)


def pytest_unconfigure(config):
    if _fixture_directory is not None:
        os.environ.pop("SEMS_CASE_PATH", None)
        _fixture_directory.cleanup()
