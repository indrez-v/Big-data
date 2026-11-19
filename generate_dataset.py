import os, random
import pandas as pd

CATEGORIES = {
    "Electronics": [("Wireless Earbuds Pro", 2499, .80), ("Smart Watch X2", 3999, .65), ("Bluetooth Speaker Mini", 1799, .55),
                    ("Power Bank 20000mAh", 1499, .85), ("Budget Smartphone S1", 8999, .35), ("Gaming Mouse RGB", 999, .70)],
    "Fashion": [("Classic Denim Jacket", 2299, .70), ("Running Shoes Air", 3499, .82), ("Cotton Kurta Set", 1299, .60),
                ("Leather Wallet", 799, .75), ("Fast-Fashion Hoodie", 999, .30), ("Sports Cap", 399, .58)],
    "Home & Kitchen": [("Non-Stick Cookware Set", 2799, .72), ("Air Fryer 4L", 5499, .84), ("Electric Kettle", 1199, .60),
                       ("Mixer Grinder 750W", 3299, .50), ("LED Study Lamp", 699, .78), ("Vacuum Flask 1L", 899, .33)],
    "Books": [("Data Science Handbook", 699, .85), ("Mystery Novel Collection", 549, .72), ("Python for Beginners", 449, .80),
              ("Self-Growth Guide", 399, .55), ("History Illustrated", 899, .68), ("Cheap Reprint Classics", 199, .30)],
    "Beauty": [("Vitamin C Serum", 599, .78), ("Herbal Shampoo", 349, .65), ("Matte Lipstick Set", 799, .60),
               ("Sunscreen SPF50", 449, .82), ("Hair Oil Natural", 299, .52), ("Budget Face Cream", 199, .32)],
}

GENERIC = ["delivery", "packaging", "price", "quality", "support"]
CAT_ASPECTS = {
    "Electronics": ["battery", "sound", "performance"],
    "Fashion": ["size", "fabric", "color"],
    "Home & Kitchen": ["durability", "cleaning", "noise"],
    "Books": ["story", "print"],
    "Beauty": ["skin", "smell", "results"],
}

TEXT = {
    "delivery": (["Delivery was fast and the item arrived early.", "Arrived on time, shipping was quick."],
                 ["Delivery was very late and the tracking was useless.", "Shipping took two weeks, really disappointed with the delivery."]),
    "packaging": (["Packaging was neat and secure.", "Well packed, the box was in perfect condition."],
                  ["The box was damaged and the packaging was poor.", "Packaging was careless, the item came in a dented box."]),
    "price": (["Great value for the price.", "Very affordable and worth every penny of the money."],
              ["Overpriced for what you get.", "Too expensive, not worth the money."]),
    "quality": (["Excellent quality, feels premium.", "Really good quality and well made."],
                ["Poor quality, it feels cheap and flimsy.", "Stopped working after a week, very poor quality."]),
    "support": (["Customer service was helpful and replied quickly.", "Support team solved my issue right away."],
                ["Customer support was rude and never solved my issue.", "No reply from customer service and the refund took forever."]),
    "battery": (["Battery life is excellent, lasts all day."], ["Battery drains very fast, terrible battery life."]),
    "sound": (["Sound quality is crisp and clear."], ["Sound is muffled and the audio keeps cutting out."]),
    "performance": (["Performance is smooth and responsive."], ["Performance is laggy and it keeps hanging."]),
    "size": (["Fits perfectly and true to size."], ["Size is wrong, the fit is very tight."]),
    "fabric": (["Fabric is soft and comfortable."], ["Fabric feels rough and the stitching came apart."]),
    "color": (["Color looks exactly like the photos."], ["Color faded after the first wash."]),
    "durability": (["Very durable and built to last."], ["Not durable, it broke within a month."]),
    "cleaning": (["Easy to clean and maintain."], ["Hard to clean and the coating peeled off."]),
    "noise": (["Runs quietly."], ["Very noisy while working."]),
    "story": (["The story is engaging and well written."], ["The content is boring and poorly explained."]),
    "print": (["Print quality is sharp and the paper is good."], ["Print quality is bad and the pages are loose."]),
    "skin": (["Skin feels soft and glowing, no irritation."], ["It caused skin irritation and breakouts."]),
    "smell": (["Smells lovely and fresh."], ["The smell is strong and unpleasant."]),
    "results": (["Saw visible results in two weeks."], ["No visible results even after a month."]),
}

NEUTRAL = ["It is okay, nothing special.", "Average product, does the job.", "Works as described, neither good nor bad.",
           "Decent but not amazing.", "It is fine, I expected a bit more.", "Mixed feelings, some things are good and some are not."]

FIRST = ["Aarav", "Priya", "Karthik", "Divya", "Rahul", "Sneha", "Arjun", "Meena", "Vikram", "Anjali", "Suresh", "Lakshmi",
         "Rohit", "Kavya", "Manoj", "Nisha", "Ajay", "Pooja", "Naveen", "Swathi"]
LAST = ["Kumar", "Raj", "Iyer", "Nair", "Reddy", "Sharma", "Pillai", "Das", "Menon", "Singh"]


def make_text(cat, rating, rng):
    p_pos = {5: .97, 4: .85, 3: .5, 2: .15, 1: .04}[rating]
    aspects = rng.sample(GENERIC + CAT_ASPECTS[cat], rng.randint(1, 3))
    parts = []
    for a in aspects:
        pos, neg = TEXT[a]
        parts.append(rng.choice(pos if rng.random() < p_pos else neg))
    if rating == 3:
        parts.insert(rng.randint(0, len(parts)), rng.choice(NEUTRAL))
    return " ".join(parts)


def main(n_users=100, seed=42):
    rng = random.Random(seed)
    prods, n = [], 1
    for cat, items in CATEGORIES.items():
        for name, price, q in items:
            prods.append(dict(product_id=f"P{n:03d}", product_name=name, category=cat, price=price, quality=q))
            n += 1

    rows, rid = [], 1
    for i in range(1, n_users + 1):
        uid, uname = f"U{i:03d}", f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        favs = rng.sample(list(CATEGORIES), rng.randint(1, 2))
        pool = prods[:]
        weights = [4 if p["category"] in favs else 1 for p in pool]
        for _ in range(rng.randint(6, 12)):
            k = rng.choices(range(len(pool)), weights)[0]
            p = pool.pop(k); weights.pop(k)
            aff = 1.0 if p["category"] in favs else 0.3
            score = p["quality"] * 0.75 + aff * 0.25 + rng.gauss(0, 0.13)
            rating = int(min(5, max(1, round(1 + 4 * score))))
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=rng.randint(0, 730))
            rows.append(dict(review_id=rid, user_id=uid, user_name=uname, product_id=p["product_id"],
                             product_name=p["product_name"], category=p["category"], price=p["price"],
                             rating=rating, review_text=make_text(p["category"], rating, rng),
                             review_date=date.strftime("%Y-%m-%d")))
            rid += 1

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), exist_ok=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reviews.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Dataset created: {path} ({len(rows)} reviews)")


if __name__ == "__main__":
    main()