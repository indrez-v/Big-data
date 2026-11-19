import os, re, json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "reviews.csv")
if not os.path.exists(DATA):
    import generate_dataset
    generate_dataset.main()

app = Flask(__name__)


def conv(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def J(o):
    return jsonify(json.loads(json.dumps(o, default=conv)))


# ------------------------------------------------------------------ DATA
df = pd.read_csv(DATA)
df["sentiment"] = df.rating.apply(lambda r: "positive" if r >= 4 else "neutral" if r == 3 else "negative")


def clean(t):
    t = re.sub(r"[^a-z\s']", " ", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()


df["clean"] = df.review_text.apply(clean)
LABELS = ["negative", "neutral", "positive"]

# ------------------------------------------------------------------ SENTIMENT MODELS
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced"),
    "Naive Bayes": MultinomialNB(),
}


def make_pipe(clf):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)), ("clf", clf)])


Xtr, Xte, ytr, yte = train_test_split(df.clean, df.sentiment, test_size=0.2, stratify=df.sentiment, random_state=42)
comparison, confusions, reports, trained = [], {}, {}, {}
for name, clf in MODELS.items():
    pipe = make_pipe(clf).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    comparison.append(dict(model=name, accuracy=accuracy_score(yte, pred), precision=p, recall=r, f1=f))
    confusions[name] = confusion_matrix(yte, pred, labels=LABELS).tolist()
    reports[name] = classification_report(yte, pred, labels=LABELS, output_dict=True, zero_division=0)
    trained[name] = pipe

DEPLOYED = "Logistic Regression"   # gives probabilities + interpretable features
model = trained[DEPLOYED]
cv_scores = cross_val_score(make_pipe(MODELS[DEPLOYED]), df.clean, df.sentiment, cv=5, scoring="f1_macro")

vec, lr = model.named_steps["tfidf"], model.named_steps["clf"]
fnames = np.array(vec.get_feature_names_out())
top_features = {c: fnames[np.argsort(lr.coef_[i])[-10:][::-1]].tolist() for i, c in enumerate(lr.classes_)}

proba = model.predict_proba(df.clean)
cls = list(model.classes_)
df["pred_sentiment"] = model.classes_[proba.argmax(1)]
df["sent_score"] = proba[:, cls.index("positive")] - proba[:, cls.index("negative")]

# ------------------------------------------------------------------ COMPLAINTS / ASPECTS
ASPECTS = {
    "Delivery": ["delivery", "shipping", "late", "tracking"], "Packaging": ["packaging", "box"],
    "Price": ["overpriced", "expensive", "money"], "Quality": ["quality", "cheap", "flimsy", "stopped working"],
    "Customer Support": ["support", "customer service", "refund"], "Battery": ["battery"],
    "Sound": ["sound", "audio"], "Performance": ["laggy", "hanging", "performance"], "Size / Fit": ["size", "fit"],
    "Fabric": ["fabric", "stitching"], "Color": ["color"], "Durability": ["durable", "broke"],
    "Cleaning": ["clean", "peeled"], "Noise": ["noisy"], "Content": ["boring", "content"],
    "Print": ["print", "pages"], "Skin Reaction": ["irritation", "breakouts"], "Smell": ["smell"],
    "No Results": ["no visible results"],
}


def has(text, kws):
    return any(re.search(r"\b" + re.escape(k), text) for k in kws)


def aspect_counts(texts):
    out = {a: int(sum(has(t, k) for t in texts)) for a, k in ASPECTS.items()}
    return {a: c for a, c in sorted(out.items(), key=lambda x: -x[1]) if c > 0}


neg = df[df.sentiment == "negative"]
ASPECT_COUNTS = aspect_counts(neg.clean)
cv_bi = CountVectorizer(ngram_range=(2, 2), stop_words="english", min_df=2)
M = cv_bi.fit_transform(neg.clean)
freq = np.asarray(M.sum(0)).ravel()
BIGRAMS = [dict(phrase=cv_bi.get_feature_names_out()[i], count=int(freq[i])) for i in freq.argsort()[::-1][:12]]

prod_rows = []
for (pid, pname, cat), g in df.groupby(["product_id", "product_name", "category"]):
    ng = g[g.sentiment == "negative"]
    ac = aspect_counts(ng.clean)
    prod_rows.append(dict(product_id=pid, product_name=pname, category=cat, n=len(g), avg_rating=g.rating.mean(),
                          neg_pct=(g.sentiment == "negative").mean() * 100,
                          pos_pct=(g.pred_sentiment == "positive").mean() * 100,
                          top_issue=next(iter(ac), "-")))
PSTATS = {r["product_id"]: r for r in prod_rows}
PINFO = {r["product_id"]: dict(name=r["product_name"], category=r["category"]) for r in prod_rows}
PRICE = df.drop_duplicates("product_id").set_index("product_id").price.to_dict()


def predict_text(text):
    x = clean(text)
    p = model.predict_proba([x])[0]
    probs = {c: float(p[i]) for i, c in enumerate(model.classes_)}
    label = max(probs, key=probs.get)
    aspects = []
    for s in re.split(r"[.!?]+", text):
        if not s.strip(): continue
        sl = clean(s)
        lab = model.predict([sl])[0]
        for a, kws in ASPECTS.items():
            if has(sl, kws):
                aspects.append(dict(aspect=a, sentiment=lab, sentence=s.strip()))
    return dict(label=label, confidence=probs[label], probs=probs, aspects=aspects)


# ------------------------------------------------------------------ RECOMMENDER
class Recommender:
    """Hybrid = 0.5 * item-based CF (cosine) + 0.3 * category preference + 0.2 * review-sentiment score."""

    def __init__(self, train, users, products):
        self.users, self.products = users, products
        self.uidx = {u: i for i, u in enumerate(users)}
        self.pidx = {p: i for i, p in enumerate(products)}
        nu, npd = len(users), len(products)
        R = np.zeros((nu, npd))
        for r in train.itertuples():
            R[self.uidx[r.user_id], self.pidx[r.product_id]] = r.rating
        self.R = R
        mask = (R > 0).astype(float)
        g = R[R > 0].mean()
        cnt = mask.sum(0)
        self.item_mean = np.where(cnt > 0, R.sum(0) / np.maximum(cnt, 1), g)
        sim = cosine_similarity(R.T)
        np.fill_diagonal(sim, 0)
        self.sim = sim
        num, den = R @ sim, mask @ sim
        self.cf = np.where(den > 1e-9, num / np.maximum(den, 1e-9), self.item_mean)   # predicted rating (1-5)
        pop = (R.sum(0) + 3 * g) / (cnt + 3) / 5                                       # Bayesian avg popularity
        s = train.groupby("product_id").sent_score.mean()
        self.sent = np.array([(s.get(p, 0.0) + 1) / 2 for p in products])
        cats = sorted({PINFO[p]["category"] for p in products})
        cidx = {c: i for i, c in enumerate(cats)}
        cat_of = np.array([cidx[PINFO[p]["category"]] for p in products])
        pref = np.zeros((nu, len(cats)))
        for u in range(nu):
            for i in np.where(R[u] > 0)[0]:
                pref[u, cat_of[i]] += R[u, i] / 5
        pref = pref / (pref.max(1, keepdims=True) + 1e-9)
        self.pref_item = pref[:, cat_of]
        self.scores = {
            "cf": self.cf / 5,
            "popularity": np.tile(pop, (nu, 1)),
            "hybrid": 0.5 * self.cf / 5 + 0.3 * self.pref_item + 0.2 * self.sent[None, :],
        }

    def recommend(self, uid, k=5):
        u = self.uidx[uid]
        s = self.scores["hybrid"][u].copy()
        s[self.R[u] > 0] = -np.inf
        liked = np.where(self.R[u] >= 4)[0]
        out = []
        for i in np.argsort(-s)[:k]:
            pid = self.products[i]
            reasons = []
            if liked.size:
                j = liked[np.argmax(self.sim[i, liked])]
                if self.sim[i, j] > 0.05:
                    reasons.append(f"Customers who liked \"{PINFO[self.products[j]]['name']}\" (you rated it {int(self.R[u, j])}★) also liked this")
            if self.pref_item[u, i] >= 0.8:
                reasons.append(f"Matches your interest in {PINFO[pid]['category']}")
            if PSTATS[pid]["pos_pct"] >= 70:
                reasons.append(f"{PSTATS[pid]['pos_pct']:.0f}% of review text is positive")
            out.append(dict(product_id=pid, name=PINFO[pid]["name"], category=PINFO[pid]["category"], price=PRICE[pid],
                            score=float(s[i]), predicted_rating=float(self.cf[u, i]),
                            avg_rating=PSTATS[pid]["avg_rating"], pos_pct=PSTATS[pid]["pos_pct"], reasons=reasons))
        return out


USERS = sorted(df.user_id.unique())
PRODUCTS = sorted(df.product_id.unique())
REC = Recommender(df, USERS, PRODUCTS)


def evaluate_recsys(K=5, seed=7):
    rng = np.random.RandomState(seed)
    test_idx = []
    for _, g in df.groupby("user_id"):
        if len(g) >= 5:
            test_idx += list(rng.choice(g.index, max(1, int(round(0.2 * len(g)))), replace=False))
    test, train = df.loc[test_idx], df.drop(test_idx)
    rec = Recommender(train, USERS, PRODUCTS)

    actual = test.rating.values
    pred = np.array([rec.cf[rec.uidx[r.user_id], rec.pidx[r.product_id]] for r in test.itertuples()])
    base = np.array([rec.item_mean[rec.pidx[p]] for p in test.product_id])
    rating = dict(
        cf_rmse=np.sqrt(np.mean((pred - actual) ** 2)), cf_mae=np.mean(np.abs(pred - actual)),
        base_rmse=np.sqrt(np.mean((base - actual) ** 2)), base_mae=np.mean(np.abs(base - actual)))

    ranking, n_users = {}, 0
    for mode in ["hybrid", "cf", "popularity"]:
        P, Rc, H, N = [], [], [], []
        for uid, g in test.groupby("user_id"):
            u = rec.uidx[uid]
            rel = {rec.pidx[p] for p in g[g.rating >= 4].product_id}
            if not rel: continue
            s = rec.scores[mode][u].copy()
            s[rec.R[u] > 0] = -np.inf
            top = [int(i) for i in np.argsort(-s)[:K]]
            hits = len(rel & set(top))
            dcg = sum(1 / np.log2(r + 2) for r, i in enumerate(top) if i in rel)
            idcg = sum(1 / np.log2(r + 2) for r in range(min(len(rel), K)))
            P.append(hits / K); Rc.append(hits / len(rel)); H.append(1 if hits else 0); N.append(dcg / idcg)
        n_users = len(P)
        ranking[mode] = dict(precision=np.mean(P), recall=np.mean(Rc), hit_rate=np.mean(H), ndcg=np.mean(N))
    return dict(k=K, users_evaluated=n_users, test_size=len(test), train_size=len(train), rating=rating, ranking=ranking)


RECSYS_EVAL = evaluate_recsys()
print("Model ready ->", DEPLOYED, "| reviews:", len(df))


# ------------------------------------------------------------------ ROUTES
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def overview():
    cat = df.groupby(["category", "sentiment"]).size().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
    top = sorted([r for r in prod_rows if r["n"] >= 5], key=lambda r: -r["avg_rating"])[:8]
    return J(dict(
        kpis=dict(reviews=len(df), customers=df.user_id.nunique(), products=df.product_id.nunique(), avg_rating=df.rating.mean()),
        sentiment=df.sentiment.value_counts().reindex(LABELS, fill_value=0).to_dict(),
        ratings=df.rating.value_counts().sort_index().to_dict(),
        by_category=[dict(category=c, **cat.loc[c].to_dict()) for c in cat.index],
        top_products=top))


@app.route("/api/predict", methods=["POST"])
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return J(dict(error="Please enter review text"))
    return J(predict_text(text))


@app.route("/api/reviews")
def reviews():
    d = df
    s, q = request.args.get("sentiment", ""), request.args.get("q", "").lower().strip()
    if s: d = d[d.sentiment == s]
    if q: d = d[d.review_text.str.lower().str.contains(q, regex=False) | d.product_name.str.lower().str.contains(q, regex=False)]
    d = d.head(40)
    return J(d[["user_name", "product_name", "rating", "review_text", "sentiment", "pred_sentiment"]].to_dict("records"))


@app.route("/api/complaints")
def complaints():
    worst = sorted([r for r in prod_rows if r["n"] >= 5], key=lambda r: -r["neg_pct"])[:8]
    return J(dict(aspects=[dict(aspect=a, count=c) for a, c in ASPECT_COUNTS.items()][:12], bigrams=BIGRAMS,
                  products=worst, features=top_features, negative_reviews=len(neg)))


@app.route("/api/users")
def users():
    return J(df.drop_duplicates("user_id")[["user_id", "user_name"]].sort_values("user_id").to_dict("records"))


@app.route("/api/recommend/<uid>")
def recommend(uid):
    if uid not in REC.uidx:
        return J(dict(error="Unknown user"))
    k = int(request.args.get("k", 5))
    h = df[df.user_id == uid]
    share = (h.category.value_counts(normalize=True) * 100).round(0)
    return J(dict(
        user=dict(user_id=uid, name=h.user_name.iloc[0], purchases=len(h), avg_rating=h.rating.mean()),
        favourites=[dict(category=c, share=v) for c, v in share.items()],
        history=h.sort_values("rating", ascending=False)[["product_name", "category", "rating", "review_text", "sentiment"]].to_dict("records"),
        recommendations=REC.recommend(uid, k)))


@app.route("/api/evaluation")
def evaluation():
    return J(dict(
        sentiment=dict(comparison=comparison, deployed=DEPLOYED, labels=LABELS, confusion=confusions[DEPLOYED],
                       report=reports[DEPLOYED], cv_mean=cv_scores.mean(), cv_std=cv_scores.std(),
                       train_size=len(Xtr), test_size=len(Xte)),
        recsys=RECSYS_EVAL))


if __name__ == "__main__":
    app.run(debug=False, port=5000)