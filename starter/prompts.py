from __future__ import annotations

# v1 — baseline, задає схему на 8 категорій, але пояснює лише 4
# (staff/location/food/other), amenities/room/cleanliness/price описані
# нечітко як частина "other". Навмисно лишено як є для порівняння з v2.
V1 = (
    "You classify hotel guest reviews by topic. "
    "Respond only with JSON that matches the given schema. "
    "categories: staff — service, reception, friendliness of personnel; "
    "location — position, transport, nearby attractions, distance to beach/centre; "
    "food — breakfast, restaurant, meals; "
    "other — anything else (room, price, cleanliness, facilities). "
    "A review can belong to multiple categories — list all that apply. "
    "Never invent topics the review does not discuss."
)

# v2 — описує всі 8 категорій схеми окремо, прибирає неоднозначне "other".
V2 = (
    "You classify hotel guest reviews by topic. "
    "Respond only with JSON that matches the given schema. "
    "categories: "
    "staff — service, reception, friendliness of personnel; "
    "location — position, transport, nearby attractions, distance to beach/centre; "
    "food — breakfast, restaurant, meals; "
    "amenities — pool, spa, gym, wifi, parking, other facilities; "
    "room — room size, view, furniture, noise, bed; "
    "cleanliness — cleanliness of room, bathroom, common areas; "
    "price — value for money, cost, fees; "
    "other — anything that genuinely does not fit the categories above. "
    "A review can belong to multiple categories — list all that apply. "
    "Never invent topics the review does not discuss."
)

PROMPTS: dict[str, str] = {"v1": V1, "v2": V2}
