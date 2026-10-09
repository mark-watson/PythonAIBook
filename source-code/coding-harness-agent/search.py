# search.py -- Brave Search and Exa AI search backends
# Python port of search.rkt (originally py-coding-agent/search.py).
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
# See LICENSE file for details

import os

import requests

EXA_ENDPOINT = "https://api.exa.ai/search"


# ---------------------------------------------------------------------------
# Brave Search
# Returns list of (url, title, description)

def brave_search(query, num_results=5):
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        raise RuntimeError("brave-search: BRAVE_SEARCH_API_KEY environment variable not set")
    url = ("https://api.search.brave.com/res/v1/web/search?q={}&count={}"
           .format(requests.utils.quote(query), num_results))
    headers = {
        "X-Subscription-Token": api_key,
        "content-type": "application/json",
        "accept": "application/json",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    data = resp.json()
    web = data.get("web", {})
    results = web.get("results", [])
    return [(r.get("url", ""), r.get("title", ""), r.get("description", ""))
            for r in results]


# ---------------------------------------------------------------------------
# Exa AI Search
# Returns list of (url, title, highlight)

def exa_search(query, num_results=5):
    api_key = os.environ.get("EXA_SEARCH_API_KEY", "")
    if not api_key:
        raise RuntimeError("exa-search: EXA_SEARCH_API_KEY environment variable not set")
    payload = {
        "query": query,
        "type": "auto",
        "numResults": num_results,
        "contents": {"highlights": True},
    }
    headers = {
        "content-type": "application/json",
        "authorization": "Bearer " + api_key,
    }
    resp = requests.post(EXA_ENDPOINT, headers=headers, json=payload, timeout=30)
    data = resp.json()
    results = data.get("results", [])
    out = []
    for r in results:
        hl = r.get("highlights") or []
        out.append((r.get("url", ""), r.get("title", ""),
                    hl[0] if isinstance(hl, list) and hl else ""))
    return out
