from ddgs import DDGS

def web_search(query, max_results=10):
    results_list = []

    with DDGS() as ddgs:
        results = ddgs.text(
            query,
            region="wt-wt",        # 🌍 глобальный поиск
            safesearch="off",
            timelimit="y",         # 📅 за последний год
            max_results=max_results
        )

        for r in results:
            title = r.get("title")
            body = r.get("body")
            href = r.get("href")

            if not title or not body or not href:
                continue

            lowered = (title + body).lower()
            if any(x in lowered for x in ["login", "sign in", "cookie", "privacy", "terms"]):
                continue

            results_list.append({
                "title": title,
                "description": body,
                "url": href
            })

    return results_list
