# -*- coding: utf-8 -*-
"""
Jev vs LLM構造化出力 vs ルールベース を同一入力で比較する評価スクリプト。

使い方:
  pip install requests anthropic openai
  export TYPESAFE_API_KEY=...        # TypeSafe API直結
  # または Cloudflare Workers AI 経由:
  export CF_ACCOUNT_ID=... CF_API_TOKEN=...
  export ANTHROPIC_API_KEY=...       # LLMベースライン（任意）
  export OPENAI_API_KEY=...          # LLMベースライン（任意）

  python run_eval.py --backends jev rule claude            # 実行
  python run_eval.py --backends jev --limit 5 --dry-run    # 接続確認
  python run_eval.py --summarize results.jsonl             # 集計だけやり直す

出力:
  results.jsonl  … 1行＝1件×1バックエンドの生結果（再集計可能）
  summary.md     … 記事に貼れる集計表
"""
import argparse, csv, json, os, statistics, sys, time
from collections import defaultdict

DEPTS = ["billing", "technical", "sales", "other"]
CRITERIA_DEPT = {
    "billing": "請求書・支払い・返金・領収書・料金の計算間違いに関する内容",
    "technical": "ログイン・エラー・不具合・設定・API・セキュリティなど製品の動作に関する内容",
    "sales": "見積もり・契約・プラン変更・解約・デモ・導入検討・料金プランの相談",
    "other": "上記いずれにも当てはまらない。挨拶のみ、採用、広報、スパム、内容不明など",
}
Q_URGENT = "この問い合わせは当日中に対応しないと顧客に実害（業務停止・金銭損失・情報漏えい）が出るか"
Q_HUMAN = "この問い合わせは自動応答ではなく、人（担当者・責任者）が直接対応すべきか（法務・補償・クレーム・情報漏えい・内容不明など）"

# 料金（2026-09時点の公表値。変わったらここを直す）
PRICE = {
    "jev":    {"in": 0.042,  "out": 0.0},     # USD / 1M tokens
    "claude": {"in": 0.80,   "out": 4.0},     # ★使うモデルの単価に合わせて修正
    "openai": {"in": 0.15,   "out": 0.60},    # ★同上
    "rule":   {"in": 0.0,    "out": 0.0},
}

# ----------------------------------------------------------------------
# Backends: それぞれ (dept, urgent(bool), human(bool), conf: dict, usage: dict) を返す
# ----------------------------------------------------------------------

def call_jev(text):
    import requests
    payload = {
        "model": os.environ.get("JEV_MODEL", "jev-latest"),
        "state": text,
        "questions": {
            "department": {"type": "choice", "instructions": "この問い合わせを担当すべき部署", "criteria": CRITERIA_DEPT},
            "is_urgent": {"type": "noul", "instructions": Q_URGENT,
                          "criteria": {"true": "当日中の対応が必要", "false": "通常の対応期限で問題ない"}},
            "needs_human": {"type": "noul", "instructions": Q_HUMAN,
                            "criteria": {"true": "人が直接対応すべき", "false": "自動応答・定型対応で問題ない"}},
        },
    }
    if os.environ.get("TYPESAFE_API_KEY"):
        r = requests.post("https://api.typesafe.ai/v1/systemone",
                          headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"},
                          json=payload, timeout=30)
    elif os.environ.get("CF_ACCOUNT_ID"):
        # Cloudflare Workers AI: {"model": "typesafe/jev", "input": {state, questions}}
        cf_payload = {"model": "typesafe/jev", "input": {"state": payload["state"], "questions": payload["questions"]}}
        r = requests.post(f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CF_ACCOUNT_ID']}/ai/run",
                          headers={"Authorization": f"Bearer {os.environ['CF_API_TOKEN']}"},
                          json=cf_payload, timeout=30)
    else:
        raise RuntimeError("TYPESAFE_API_KEY か CF_ACCOUNT_ID/CF_API_TOKEN を設定してください")
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    if "result" in data and "answers" not in data:   # Cloudflare REST envelope
        if not data.get("success", True):
            raise RuntimeError(f"Cloudflare error: {data.get('errors')}")
        data = data["result"]
    ans = data["answers"]
    d = ans["department"]; u = ans["is_urgent"]; h = ans["needs_human"]
    pu = u.get("noul", u.get("probability")); ph = h.get("noul", h.get("probability"))
    conf = {"dept": d.get("confidence"), "dept_probs": d.get("probabilities"), "urgent": pu, "human": ph,
            "model": data.get("model")}
    return d["choice"], pu >= 0.5, ph >= 0.5, conf, data.get("usage", {})

LLM_SYSTEM = f"""あなたはカスタマーサポートの一次振り分け担当です。問い合わせ文を読み、必ずJSONだけを返してください。
{{"department": "billing|technical|sales|other", "is_urgent": true|false, "needs_human": true|false, "confidence": 0.0-1.0}}
部署の定義:
{json.dumps(CRITERIA_DEPT, ensure_ascii=False, indent=1)}
is_urgent: {Q_URGENT}
needs_human: {Q_HUMAN}"""

def call_claude(text):
    import anthropic
    client = anthropic.Anthropic()
    model = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")  # ★実行時に最新の軽量モデル名へ
    msg = client.messages.create(model=model, max_tokens=200, system=LLM_SYSTEM,
                                 messages=[{"role": "user", "content": text}])
    raw = msg.content[0].text.strip().strip("`").removeprefix("json").strip()
    j = json.loads(raw)
    usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
    return j["department"], bool(j["is_urgent"]), bool(j["needs_human"]), {"dept": j.get("confidence")}, usage

def call_openai(text):
    from openai import OpenAI
    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")  # ★実行時に最新の軽量モデル名へ
    r = client.chat.completions.create(model=model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": text}])
    j = json.loads(r.choices[0].message.content)
    usage = {"input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens}
    return j["department"], bool(j["is_urgent"]), bool(j["needs_human"]), {"dept": j.get("confidence")}, usage

RULES = {
    "billing": ["請求", "支払", "返金", "領収書", "引き落と", "振込", "インボイス", "invoice", "账单"],
    "technical": ["ログイン", "エラー", "不具合", "API", "認証", "通知", "ダウン", "落ち", "動きません", "動かない", "500", "503", "login", "アカウント", "権限", "バックアップ", "設定"],
    "sales": ["見積", "契約", "プラン", "解約", "デモ", "導入", "トライアル", "割引", "devis", "plan"],
}
URGENT_WORDS = ["至急", "急ぎ", "今日中", "本日中", "ダウン", "止まって", "漏", "不正", "削除され", "停止", "二重", "2回", "倍になって", "急ぎです"]
NEG_WORDS = ["急ぎではありません", "急ぎません", "緊急ではない", "急ぎでなくて", "急ぎでは"]
HUMAN_WORDS = ["責任者", "法的", "法務", "補償", "漏", "報道", "退会", "不適切", "いい加減", "何度言えば", "返金"]

def call_rule(text):
    t = text.lower()
    scores = {d: sum(t.count(w.lower()) for w in ws) for d, ws in RULES.items()}
    dept = max(scores, key=scores.get) if max(scores.values()) > 0 else "other"
    urgent = any(w in text for w in URGENT_WORDS) and not any(w in text for w in NEG_WORDS)
    human = any(w in text for w in HUMAN_WORDS) or len(text.strip()) <= 6
    return dept, urgent, human, {"dept": None}, {"input_tokens": 0, "output_tokens": 0}

BACKENDS = {"jev": call_jev, "claude": call_claude, "openai": call_openai, "rule": call_rule}

# ----------------------------------------------------------------------

def run(args):
    rows = list(csv.DictReader(open(args.dataset, encoding="utf-8")))
    if args.limit: rows = rows[:args.limit]
    out = open(args.out, "a", encoding="utf-8")
    for b in args.backends:
        fn = BACKENDS[b]
        for i, r in enumerate(rows, 1):
            rec = {"backend": b, "id": r["id"], "category": r["category"],
                   "gold_dept": r["dept"], "gold_urgent": r["urgent"] == "1", "gold_human": r["needs_human"] == "1"}
            for attempt in range(3):
                try:
                    t0 = time.perf_counter()
                    dept, urgent, human, conf, usage = fn(r["text"])
                    rec.update({"latency_ms": (time.perf_counter() - t0) * 1000, "pred_dept": dept,
                                "pred_urgent": urgent, "pred_human": human, "conf": conf, "usage": usage, "error": None})
                    break
                except Exception as e:
                    rec.update({"error": repr(e)}); time.sleep(2 ** attempt)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            print(f"[{b}] {i}/{len(rows)} {r['id']} -> {rec.get('pred_dept')} u={rec.get('pred_urgent')} h={rec.get('pred_human')} {rec.get('latency_ms', 0):.0f}ms {rec.get('error') or ''}")
            if args.dry_run and i >= 1: break
    out.close()

def pct(a, b): return f"{100*a/b:.1f}%" if b else "-"

def summarize(path, low_conf=0.7):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    by = defaultdict(list)
    for r in recs: by[r["backend"]].append(r)
    lines = ["| 指標 | " + " | ".join(by) + " |", "|---|" + "---|" * len(by)]
    def row(name, f): lines.append(f"| {name} | " + " | ".join(f(v) for v in by.values()) + " |")
    ok = lambda v: [r for r in v if not r.get("error")]
    row("件数（成功/合計）", lambda v: f"{len(ok(v))}/{len(v)}")
    row("部署 正答率", lambda v: pct(sum(r["pred_dept"] == r["gold_dept"] for r in ok(v)), len(ok(v))))
    row("緊急度 正答率", lambda v: pct(sum(r["pred_urgent"] == r["gold_urgent"] for r in ok(v)), len(ok(v))))
    row("要人対応 正答率", lambda v: pct(sum(r["pred_human"] == r["gold_human"] for r in ok(v)), len(ok(v))))
    # 重大誤り: 緊急を非緊急と判定 / 要人対応を自動と判定
    row("重大誤り: 緊急の見逃し", lambda v: f"{sum(r['gold_urgent'] and not r['pred_urgent'] for r in ok(v))}/{sum(r['gold_urgent'] for r in ok(v))}")
    row("重大誤り: 要人対応の見逃し", lambda v: f"{sum(r['gold_human'] and not r['pred_human'] for r in ok(v))}/{sum(r['gold_human'] for r in ok(v))}")
    for cat in ["normal", "boundary", "negation", "insufficient", "dialect", "typo", "emoji", "sarcasm", "multi", "multilingual"]:
        row(f"部署正答率 [{cat}]", lambda v, c=cat: pct(sum(r["pred_dept"] == r["gold_dept"] for r in ok(v) if r["category"] == c),
                                                     len([r for r in ok(v) if r["category"] == c])))
    def lowconf(v):
        vv = [r for r in ok(v) if r.get("conf", {}).get("dept") is not None]
        if not vv: return "-"
        low = [r for r in vv if r["conf"]["dept"] < low_conf]
        wrong_in_low = sum(r["pred_dept"] != r["gold_dept"] for r in low)
        wrong_total = sum(r["pred_dept"] != r["gold_dept"] for r in vv)
        return f"低信頼{len(low)}件、うち誤り{wrong_in_low}（全誤り{wrong_total}の{pct(wrong_in_low, wrong_total)}）"
    row(f"低信頼度(<{low_conf})での誤り検知", lowconf)
    def lat(v, q):
        xs = sorted(r["latency_ms"] for r in ok(v))
        return f"{xs[min(len(xs)-1, int(len(xs)*q))]:.0f}ms" if xs else "-"
    row("レイテンシ P50", lambda v: lat(v, 0.5)); row("レイテンシ P95", lambda v: lat(v, 0.95))
    def cost(v):
        b = v[0]["backend"]; p = PRICE.get(b, {"in": 0, "out": 0})
        tin = sum(r.get("usage", {}).get("input_tokens", 0) for r in ok(v)); tout = sum(r.get("usage", {}).get("output_tokens", 0) for r in ok(v))
        c = (tin * p["in"] + tout * p["out"]) / 1e6
        return f"${c:.5f}（1件 ${c/len(ok(v)):.6f}）" if ok(v) else "-"
    row("推定コスト（API単価のみ）", cost)
    md = "\n".join(lines)
    open("summary.md", "w", encoding="utf-8").write(md); print(md)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_ja_inquiries.csv")
    ap.add_argument("--backends", nargs="*", default=["jev", "rule"], choices=list(BACKENDS))
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize", metavar="RESULTS")
    a = ap.parse_args()
    if a.summarize: summarize(a.summarize)
    else: run(a); summarize(a.out)
