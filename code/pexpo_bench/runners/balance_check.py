"""Check the provider credentials and available balance for the selected models."""
from __future__ import annotations

import os
import sys
import requests
from typing import Optional


def _check_deepseek() -> dict:
    key = os.environ.get("deepseek_API_KEY")
    if not key: return {"endpoint": "deepseek", "status": "NO_KEY"}
    try:
        r = requests.get("https://api.deepseek.com/user/balance",
                         headers={"Authorization": f"Bearer {key}"},
                         timeout=10.0)
        if r.status_code == 200:
            data = r.json()
            # Schema: {"is_available": bool, "balance_infos": [{"currency", "total_balance", ...}]}
            info = (data.get("balance_infos") or [{}])[0]
            total = float(info.get("total_balance", 0) or 0)
            currency = info.get("currency", "USD")
            return {"endpoint": "deepseek", "status": "OK",
                    "balance": total, "currency": currency,
                    "available": data.get("is_available", True)}
        return {"endpoint": "deepseek", "status": f"HTTP {r.status_code}",
                "body": "[REDACTED_RESPONSE_BODY]"}
    except Exception as e:
        return {"endpoint": "deepseek", "status": f"ERROR: {type(e).__name__}",
                "msg": type(e).__name__}


def _check_openai_native() -> dict:
    key = os.environ.get("OPENAI_API_KEY_NATIVE")
    if not key: return {"endpoint": "openai_native", "status": "NO_KEY"}
    # Project-scoped keys cannot query billing; just verify auth via /models
    try:
        r = requests.get("https://api.openai.com/v1/models",
                         headers={"Authorization": f"Bearer {key}"},
                         timeout=10.0)
        if r.status_code == 200:
            return {"endpoint": "openai_native", "status": "OK_AUTH",
                    "balance": "check at platform.openai.com/usage",
                    "note": "project-scoped key cannot query billing endpoint"}
        return {"endpoint": "openai_native", "status": f"HTTP {r.status_code}",
                "body": "[REDACTED_RESPONSE_BODY]"}
    except Exception as e:
        return {"endpoint": "openai_native", "status": f"ERROR: {type(e).__name__}",
                "msg": type(e).__name__}


def pre_flight_balance_check(min_required_usd: dict | None = None,
                             models: list[str] | None = None) -> bool:
    """Print balances, return True if all endpoints look safe to proceed.

    min_required_usd: optional {endpoint_name: minimum_balance_usd}.
        If a programmable endpoint reports balance below this, returns False.
    """
    print("=" * 60)
    print("API PRE-FLIGHT BALANCE CHECK")
    print("=" * 60)
    models = models or ["gpt-5.4", "deepseek-v4"]
    results = []
    if any(model.startswith("gpt-") for model in models):
        results.append(_check_openai_native())
    if "deepseek-v4" in models:
        results.append(_check_deepseek())
    all_ok = True
    for r in results:
        ep = r["endpoint"]
        st = r["status"]
        icon = "✅" if st.startswith("OK") else "⚠️" if st == "NO_KEY" else "❌"
        line = f"  {icon} {ep:18s} {st}"
        if "balance" in r:
            line += f"  balance={r['balance']}"
        print(line)
        if "note" in r: print(f"     note: {r['note']}")
        if "msg" in r: print(f"     msg:  {r['msg']}")
        # Hard fail criteria
        if not st.startswith("OK"):
            all_ok = False
        if (min_required_usd
                and ep in min_required_usd
                and isinstance(r.get("balance"), (int, float))
                and r["balance"] < min_required_usd[ep]):
            print(f"     ❌ below required {min_required_usd[ep]} USD")
            all_ok = False
    print("-" * 60)
    print(f"  {'OK to proceed' if all_ok else '⚠️ STOP — fix issues above before running'}")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(os.environ.get("PEXPO_ROOT", Path(__file__).resolve().parents[3])) / ".env")
    ok = pre_flight_balance_check()
    sys.exit(0 if ok else 1)
