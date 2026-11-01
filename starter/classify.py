from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-6-luna"

DATA = Path(__file__).resolve().parent.parent / "data" / "hotels.json"
LIMIT = 10


class ReviewClassification(BaseModel):
    categories: list[
        Literal["staff", "location", "food", "amenities", "room", "cleanliness", "price", "other"]] = Field(
        description="Topics the review talks about. Can list more than one."
    )
    summary: str = Field(description="Short summary of the review, up to 15 words.")


SYSTEM_PROMPT = (
    "You classify hotel guest reviews by topic. "
    "Respond only with JSON that matches the given schema. "
    "categories: staff — service, reception, friendliness of personnel; "
    "location — position, transport, nearby attractions, distance to beach/centre; "
    "food — breakfast, restaurant, meals; "
    "other — anything else (room, price, cleanliness, facilities). "
    "A review can belong to multiple categories — list all that apply. "
    "Never invent topics the review does not discuss."
)


def classify(client: OpenAI, text: str) -> dict:
    start = time.perf_counter()
    schema = ReviewClassification.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(schema.get("properties", {}).keys())
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "review",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
    except OpenAIError as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "seconds": time.perf_counter() - start,
        }

    seconds = time.perf_counter() - start
    content = r.choices[0].message.content or ""
    result = {
        "seconds": seconds,
        "input_tokens": r.usage.prompt_tokens if r.usage else 0,
        "output_tokens": r.usage.completion_tokens if r.usage else 0,
        "raw": content,
    }
    try:
        result["review"] = ReviewClassification.model_validate_json(content)
        result["ok"] = True
    except ValidationError as e:
        result["ok"] = False
        result["error"] = str(e)
    return result


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    in_p, out_p = 0.10, 0.50
    return (input_tokens * in_p + output_tokens * out_p) / 1_000_000


def main() -> None:
    if not API_KEY:
        raise SystemExit("OPENAI_API_KEY is missing in .env")

    client = OpenAI(api_key=API_KEY, timeout=30, max_retries=2)
    reviews = json.loads(DATA.read_text(encoding="utf-8"))
    reviews = reviews[:LIMIT]

    print(f"Model: {MODEL}\n")
    total_s = total_cost = 0.0
    valid = 0

    for t in reviews:
        res = classify(client, t["text"])
        total_s += res["seconds"]
        if "input_tokens" in res:
            total_cost += cost_usd(res["input_tokens"], res["output_tokens"])

        if res["ok"]:
            rv = res["review"]
            valid += 1
            print(
                f"{t['city']}/{t['hotel_name']}  {res['seconds']:.1f}s  "
                f"{', '.join(rv.categories):<20}  {t['text']}"
            )
        else:
            print(
                f"{t['city']}/{t['hotel_name']}  {res['seconds']:.1f}s  ERROR: {res['error']}"
            )

    n = len(reviews)
    print(
        f"\nFOR CHAT: {MODEL} | {n} requests | {total_s:.1f}s | ${total_cost:.6f} | "
        f"valid JSON {valid}/{n}"
    )


if __name__ == "__main__":
    main()
