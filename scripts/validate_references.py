#!/usr/bin/env python3
import argparse
import csv
import dataclasses
import json
import os
import re
import sys
import time
import typing as t
from dataclasses import dataclass

try:
    import requests
except Exception as e:
    print("This script requires the 'requests' package. Install via: pip install requests", file=sys.stderr)
    raise

try:
    import bibtexparser  # type: ignore
except Exception as e:
    print("This script requires the 'bibtexparser' package. Install via: pip install bibtexparser", file=sys.stderr)
    raise


ARXIV_ABS_RE = re.compile(r"arxiv\.org/(abs|pdf)/(?P<id>\d{4}\.\d{4,5}(v\d+)?)", re.IGNORECASE)
ARXIV_JOURNAL_RE = re.compile(r"arXiv\s+preprint\s+arXiv:(?P<id>\d{4}\.\d{4,5}(v\d+)?)", re.IGNORECASE)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def normalize_title(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def norm_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z ]+", "", name)
    return name


def extract_arxiv_id(entry: dict) -> t.Optional[str]:
    # Prefer explicit URL
    for field in ("url", "eprint", "archiveprefix", "journal"):
        val = entry.get(field)
        if not val:
            continue
        m = ARXIV_ABS_RE.search(val)
        if m:
            return m.group("id")
        m2 = ARXIV_JOURNAL_RE.search(val)
        if m2:
            return m2.group("id")
        # Some bibs have eprint like 2101.00027
        if field == "eprint":
            e = val.strip()
            if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", e):
                return e
    return None


def extract_doi(entry: dict) -> t.Optional[str]:
    if entry.get("doi"):
        m = DOI_RE.search(entry["doi"])  # allow extra text
        if m:
            return m.group(0)
    # Sometimes DOI is in url
    if entry.get("url"):
        m = DOI_RE.search(entry["url"])  # allow extra text
        if m:
            return m.group(0)
    return None


def requests_get_json(url: str, headers: dict = None, params: dict = None, timeout: int = 20) -> t.Tuple[int, t.Optional[dict]]:
    try:
        r = requests.get(url, headers=headers or {"User-Agent": "ref-validator/1.0"}, params=params, timeout=timeout)
        ct = r.headers.get("content-type", "")
        if "json" in ct:
            return r.status_code, r.json()
        return r.status_code, None
    except Exception as e:
        return 0, None


def requests_head_or_get(url: str, timeout: int = 20) -> int:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout, headers={"User-Agent": "ref-validator/1.0"})
        if r.status_code >= 400 or r.status_code == 405:
            r = requests.get(url, allow_redirects=True, timeout=timeout, headers={"User-Agent": "ref-validator/1.0"})
        return r.status_code
    except Exception:
        return 0


@dataclass
class ValidationResult:
    key: str
    entry_type: str
    title: str = ""
    year: t.Optional[int] = None
    authors: t.List[str] = dataclasses.field(default_factory=list)
    doi: t.Optional[str] = None
    arxiv_id: t.Optional[str] = None
    url: t.Optional[str] = None
    venue: t.Optional[str] = None
    status: str = "unknown"  # verified_doi | verified_arxiv | verified_url | mismatch | not_found | partial
    notes: t.List[str] = dataclasses.field(default_factory=list)


def parse_bib(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        db = bibtexparser.load(f)
    entries = {}
    for e in db.entries:
        entries[e["ID"]] = e
    return entries


def read_used_keys_from_bbl(bbl_path: str) -> t.Set[str]:
    keys: t.Set[str] = set()
    if not os.path.exists(bbl_path):
        return keys
    # The bbl contains lines like: \entry{key}{type}{}
    pat = re.compile(r"\\entry\{([^}]+)\}")
    with open(bbl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                keys.add(m.group(1))
    return keys


def validate_with_crossref(doi: str) -> t.Tuple[bool, dict]:
    url = f"https://api.crossref.org/works/{doi}"
    status, js = requests_get_json(url)
    if status == 200 and js and js.get("message"):
        return True, js["message"]
    return False, {}


def validate_with_arxiv(arxiv_id: str) -> t.Tuple[bool, dict]:
    # Use arXiv API: http://export.arxiv.org/api/query?id_list=ID
    api = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        r = requests.get(api, timeout=20, headers={"User-Agent": "ref-validator/1.0"})
        if r.status_code != 200:
            return False, {}
        text = r.text
        # Extract <title>, <author><name>, <published>
        # Take first entry only
        entry_block = re.search(r"<entry>(.*?)</entry>", text, re.DOTALL)
        if not entry_block:
            return False, {}
        entry_text = entry_block.group(1)
        title_m = re.search(r"<title>(.*?)</title>", entry_text, re.DOTALL)
        title = title_m.group(1).strip() if title_m else ""
        # Authors
        authors = [m.strip() for m in re.findall(r"<author>\s*<name>(.*?)</name>\s*</author>", entry_text, re.DOTALL)]
        pub_m = re.search(r"<published>(.*?)</published>", entry_text)
        year = None
        if pub_m:
            y = pub_m.group(1).strip()[:4]
            if y.isdigit():
                year = int(y)
        return True, {"title": title, "authors": authors, "year": year}
    except Exception:
        return False, {}


def search_arxiv_by_title(title: str) -> t.Optional[dict]:
    # Search arXiv by exact title
    q = f'ti:"{title}"'
    api = f"http://export.arxiv.org/api/query?search_query={requests.utils.quote(q)}&max_results=5"
    try:
        r = requests.get(api, timeout=20, headers={"User-Agent": "ref-validator/1.0"})
        if r.status_code != 200:
            return None
        text = r.text
        entries = re.findall(r"<entry>(.*?)</entry>", text, flags=re.DOTALL)
        for entry_text in entries:
            title_m = re.search(r"<title>(.*?)</title>", entry_text, re.DOTALL)
            cand_title = title_m.group(1).strip() if title_m else ""
            if compare_titles(title, cand_title):
                id_m = re.search(r"<id>https?://arxiv\.org/abs/([0-9.]+v\d+|[0-9.]+)</id>", entry_text)
                authors = [m.strip() for m in re.findall(r"<author>\s*<name>(.*?)</name>\s*</author>", entry_text, re.DOTALL)]
                pub_m = re.search(r"<published>(.*?)</published>", entry_text)
                year = None
                if pub_m:
                    y = pub_m.group(1).strip()[:4]
                    if y.isdigit():
                        year = int(y)
                if id_m:
                    return {"arxiv_id": id_m.group(1), "title": cand_title, "authors": authors, "year": year}
        return None
    except Exception:
        return None


def compare_titles(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return normalize_title(a) == normalize_title(b)


def compare_authors(list_a: t.List[str], list_b: t.List[str]) -> bool:
    if not list_a or not list_b:
        return False
    # Compare by last names set overlap >= 2 or >= 50%
    def last(name: str) -> str:
        parts = [p for p in re.split(r"\s+", name) if p]
        return parts[-1] if parts else name
    a_last = {norm_name(last(n)) for n in list_a}
    b_last = {norm_name(last(n)) for n in list_b}
    inter = a_last & b_last
    min_needed = max(1, min(len(a_last), len(b_last)) // 2)
    return len(inter) >= min_needed


def validate_entry(key: str, entry: dict) -> ValidationResult:
    res = ValidationResult(key=key, entry_type=entry.get("ENTRYTYPE", ""))
    res.title = entry.get("title", "").strip("{} ")
    res.venue = entry.get("journal") or entry.get("booktitle")
    res.url = entry.get("url")
    # Extract year as int if possible
    try:
        res.year = int(re.sub(r"[^0-9]", "", entry.get("year", ""))[:4]) if entry.get("year") else None
    except Exception:
        res.year = None
    # Authors
    raw_authors = entry.get("author")
    if raw_authors:
        # bibtex authors separated by ' and '
        res.authors = [a.strip() for a in raw_authors.split(" and ") if a.strip()]

    res.doi = extract_doi(entry)
    res.arxiv_id = extract_arxiv_id(entry)

    # Try DOI first
    if res.doi:
        ok, msg = validate_with_crossref(res.doi)
        if ok:
            cr_title = " ".join(msg.get("title", [])).strip()
            cr_authors = []
            for a in msg.get("author", []) or []:
                name = " ".join([p for p in [a.get("given"), a.get("family")] if p])
                if name:
                    cr_authors.append(name)
            cr_year = None
            issued = msg.get("issued", {}).get("'date-parts'", msg.get("issued", {}).get("date-parts"))
            if isinstance(issued, list) and issued and isinstance(issued[0], list) and issued[0]:
                y = issued[0][0]
                if isinstance(y, int):
                    cr_year = y
            cr_container = " ".join(msg.get("container-title", [])).strip()

            # If DOI resolves, consider the entry real. Record any field differences as notes.
            res.status = "verified_doi"
            title_match = compare_titles(res.title, cr_title)
            author_match = compare_authors(res.authors, cr_authors) if res.authors and cr_authors else True
            year_match = (res.year == cr_year) if (res.year and cr_year) else True
            if not title_match:
                res.notes.append("Title differs from Crossref")
            if not author_match:
                res.notes.append("Authors differ from Crossref")
            if not year_match:
                res.notes.append("Year differs from Crossref")
            if cr_container:
                have_venue = (res.venue or "").strip()
                if have_venue and normalize_title(have_venue) != normalize_title(cr_container):
                    res.notes.append("Venue differs from Crossref")
            return res
        else:
            res.notes.append("DOI not found in Crossref")

    # Try arXiv
    if res.arxiv_id:
        ok, meta = validate_with_arxiv(res.arxiv_id)
        if ok:
            # If arXiv ID resolves, consider the entry real. Record differences as notes.
            res.status = "verified_arxiv"
            title_match = compare_titles(res.title, meta.get("title", ""))
            author_match = compare_authors(res.authors, meta.get("authors", [])) if res.authors and meta.get("authors") else True
            year_match = (res.year == meta.get("year")) if (res.year and meta.get("year")) else True
            if not title_match:
                res.notes.append("Title differs from arXiv")
            if not author_match:
                res.notes.append("Authors differ from arXiv")
            if not year_match:
                res.notes.append("Year differs from arXiv")
            return res
        else:
            res.notes.append("arXiv ID not found")

    # If URL present, check reachability
    if res.url:
        code = requests_head_or_get(res.url)
        if code and code < 400:
            if res.status == "unknown":
                res.status = "verified_url"
            res.notes.append(f"URL reachable ({code})")
            return res
        else:
            res.notes.append("URL not reachable")

    # Try arXiv title search if nothing yet
    if res.status == "unknown" and res.title:
        found = search_arxiv_by_title(res.title)
        if found and found.get("arxiv_id"):
            res.arxiv_id = found["arxiv_id"]
            res.status = "verified_arxiv"
            # Record any differences
            if not compare_titles(res.title, found.get("title", "")):
                res.notes.append("Title differs from arXiv (search)")
            if res.year and found.get("year") and res.year != found.get("year"):
                res.notes.append("Year differs from arXiv (search)")
            return res

    # Hardcoded known URLs for common tech reports with no DOI
    if res.status == "unknown" and normalize_title(res.title) == normalize_title("Language models are unsupervised multitask learners"):
        u = "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"
        code = requests_head_or_get(u)
        if code and code < 400:
            res.url = u
            res.status = "verified_url"
            res.notes.append(f"Fallback URL reachable ({code})")
            return res

    # As a last resort, try Crossref search by title
    if res.title:
        q = res.title
        url = "https://api.crossref.org/works"
        status, js = requests_get_json(url, params={"query.bibliographic": q, "rows": 3})
        if status == 200 and js and js.get("message", {}).get("items"):
            # Pick the best by normalized title match
            items = js["message"]["items"]
            for it in items:
                cr_title = " ".join(it.get("title", [])).strip()
                if compare_titles(res.title, cr_title):
                    res.status = "partial"
                    res.notes.append("Found matching title in Crossref search")
                    it_doi = it.get("DOI")
                    if it_doi:
                        res.notes.append(f"Candidate DOI: {it_doi}")
                    break

    if res.status == "unknown":
        res.status = "not_found"
    return res


def main():
    ap = argparse.ArgumentParser(description="Validate bibliography entries against arXiv/Crossref/URLs")
    ap.add_argument("--bib", default="thesis/references.bib", help="Path to .bib file")
    ap.add_argument("--bbl", default="thesis/main.bbl", help="Optional path to .bbl file to limit to used keys")
    ap.add_argument("--output", default="thesis/reference_validation_report.json", help="Output JSON report path")
    ap.add_argument("--csv", default="thesis/reference_validation_report.csv", help="Optional CSV report path")
    ap.add_argument("--all", action="store_true", help="Validate all entries in .bib (ignore .bbl)")
    args = ap.parse_args()

    entries = parse_bib(args.bib)
    used_keys: t.Set[str] = set()
    if not args.all:
        used_keys = read_used_keys_from_bbl(args.bbl)
        if not used_keys:
            print(f"Warning: no keys read from {args.bbl}; defaulting to all entries.")
            used_keys = set(entries.keys())
    else:
        used_keys = set(entries.keys())

    results: t.List[ValidationResult] = []
    for key in sorted(used_keys):
        e = entries.get(key)
        if not e:
            vr = ValidationResult(key=key, entry_type="unknown", status="not_found")
            vr.notes.append("Key not found in .bib")
            results.append(vr)
            continue
        vr = validate_entry(key, e)
        results.append(vr)

    # Persist reports
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    # CSV summary
    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "status", "entry_type", "year", "doi", "arxiv_id", "venue", "url", "notes", "title"])
        for r in results:
            w.writerow([r.key, r.status, r.entry_type, r.year or "", r.doi or "", r.arxiv_id or "", r.venue or "", r.url or "", "; ".join(r.notes), r.title])

    # Print concise summary
    summary = {}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1
    print("Validation summary:")
    for k, v in sorted(summary.items(), key=lambda kv: kv[0]):
        print(f"  {k:>14}: {v}")
    print(f"\nJSON: {args.output}\nCSV:  {args.csv}")


if __name__ == "__main__":
    main()
