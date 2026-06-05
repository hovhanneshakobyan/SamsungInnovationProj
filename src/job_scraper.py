"""
job_scraper.py

LinkedIn  — login wall detected → clear "paste manually" message.
            Guest API attempted for direct job post URLs.
staff.am  — Content is 100% client-side rendered; requests.get() only
            returns the company "About us" section, never the JD.
            Strategy:
              1. Try __NEXT_DATA__ JSON (works if SSR is enabled for that page)
              2. Return og:description preview + a clear paste instruction
            The app.py UI shows a dedicated "Paste from staff.am" expander
            that calls parse_staff_am_paste() to extract clean JD text
            from whatever the user copies out of their browser.
Indeed    — #jobDescriptionText div.
Generic   — ranked CSS selectors → <p> tags → og:description.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_LINKEDIN_WALL = [
    "join linkedin", "sign in", "user agreement",
    "privacy policy", "cookie policy", "new to linkedin",
]

# Russian section headers staff.am uses on job pages
# Maps Russian label → English label for the output
_STAFF_AM_SECTIONS = {
    "описание вакансии":   "Job Description",
    "обязанности":         "Responsibilities",
    "требования":          "Requirements",
    "требуемый уровень":   "Level",
    "профессиональные навыки": "Required Skills",
    "условия":             "Conditions",
    "о компании":          None,   # None = stop here, it's company boilerplate
    "о нас":               None,
    "фотогалерея":         None,
    "привилегии":          None,
}

# English equivalents for en/ locale
_STAFF_AM_SECTIONS_EN = {
    "job description":     "Job Description",
    "responsibilities":    "Responsibilities",
    "requirements":        "Requirements",
    "required skills":     "Required Skills",
    "conditions":          "Conditions",
    "about the company":   None,
    "about us":            None,
    "gallery":             None,
    "benefits":            None,
}


# ── text helpers ──────────────────────────────────────────────────────────────

def _html_to_text(html_str: str) -> str:
    """Convert HTML string to clean plain text with • bullet points."""
    if not html_str:
        return ""
    if "<" not in html_str:
        return _clean_ws(html_str)

    soup = BeautifulSoup(html_str, "html.parser")

    for li in soup.find_all("li"):
        li.insert_before("\n• ")

    for tag in soup.find_all(["p", "ul", "ol", "h1", "h2", "h3", "h4"]):
        tag.append("\n")

    text = soup.get_text("")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_ws(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _hostname(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _is_linkedin_wall(text: str) -> bool:
    low = text.lower()
    return sum(1 for p in _LINKEDIN_WALL if p in low) >= 3


def _get_soup(url: str):
    try:
        r = requests.get(url, headers=_HEADERS, timeout=12)
        if r.status_code != 200:
            return None, None
        return r, BeautifulSoup(r.text, "html.parser")
    except Exception:
        return None, None


# ── Next.js __NEXT_DATA__ helpers ─────────────────────────────────────────────

def _extract_next_data(soup: BeautifulSoup) -> dict | None:
    tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if not tag or not tag.string:
        return None
    try:
        return json.loads(tag.string)
    except Exception:
        return None


_JD_KEYS = [
    "description", "jobDescription", "job_description",
    "responsibilities", "requirements", "qualifications",
    "duties", "about_job", "aboutJob", "details", "body",
    "full_description", "fullDescription",
]

_TITLE_KEYS   = ["title", "jobTitle", "name", "position"]
_SKILLS_KEYS  = ["skills", "requiredSkills", "technicalSkills"]


def _get_str(obj: dict, keys: list) -> str:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list):
            items = [str(i) for i in v if i]
            if items:
                return ", ".join(items)
    return ""


def _walk_for_value(obj, keys: list) -> str | None:
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                v = obj[k]
                if isinstance(v, str) and len(v) > 30:
                    return v
        for v in obj.values():
            r = _walk_for_value(v, keys)
            if r:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _walk_for_value(item, keys)
            if r:
                return r
    return None


def _staff_am_from_next_data(next_data: dict) -> str | None:
    page_props = next_data.get("props", {}).get("pageProps", {})

    job_obj: dict = {}
    for key in ("job", "data", "vacancy", "position", "posting", "jobPost"):
        candidate = page_props.get(key)
        if isinstance(candidate, dict):
            job_obj = candidate
            break
    if not job_obj:
        job_obj = page_props

    title = _get_str(job_obj, _TITLE_KEYS) or _get_str(page_props, _TITLE_KEYS)
    parts: list[str] = []

    for key in _JD_KEYS:
        for source in (job_obj, page_props):
            val = source.get(key)
            if isinstance(val, str) and len(val) > 50:
                cleaned = _html_to_text(val)
                if cleaned and cleaned not in parts:
                    parts.append(cleaned)
                break

    if not parts:
        found = _walk_for_value(page_props, _JD_KEYS)
        if found and len(found) > 50:
            parts.append(_html_to_text(found))

    if not parts:
        return None

    skills_raw = job_obj.get("skills") or page_props.get("skills")
    skills_str = ""
    if isinstance(skills_raw, list):
        skills_str = ", ".join(str(s) for s in skills_raw if s)
    elif isinstance(skills_raw, str):
        skills_str = skills_raw

    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("")
    lines.extend(parts)
    if skills_str:
        lines.append("")
        lines.append("Required skills: " + skills_str)

    return _clean_ws("\n".join(lines))


def _meta_og_description(soup: BeautifulSoup) -> str | None:
    for attr in [("property", "og:description"), ("name", "description")]:
        tag = soup.find("meta", {attr[0]: attr[1]})
        if tag and tag.get("content") and len(tag["content"]) > 80:
            return _clean_ws(tag["content"])
    return None


# ── staff.am paste parser (public function used by app.py) ───────────────────

def parse_staff_am_paste(raw_text: str) -> str:
    """
    Parse text that the user copied from a staff.am job page in their browser.
    Extracts Описание вакансии / Обязанности / Требования / Skills sections,
    strips the company "О компании" boilerplate, and returns clean JD text.
    Works for both Russian (/ru/) and English (/en/) locale pages.
    """
    if not raw_text or not raw_text.strip():
        return ""

    combined_sections = {**_STAFF_AM_SECTIONS, **_STAFF_AM_SECTIONS_EN}

    lines = raw_text.splitlines()
    result_sections: list[tuple[str, list[str]]] = []   # (label, [lines])
    current_label: str | None = None
    current_lines: list[str] = []
    stop = False

    for line in lines:
        stripped = line.strip()
        low = stripped.lower()

        # Check if this line is a known section header
        matched_header = None
        for header, english_label in combined_sections.items():
            if low == header or low.startswith(header + ":"):
                matched_header = (header, english_label)
                break

        if matched_header:
            # Save previous section if it has content
            if current_label is not None and current_lines:
                result_sections.append((current_label, current_lines))

            _, english_label = matched_header
            if english_label is None:
                # Hit a stop-section — everything from here is boilerplate
                stop = True
                break

            current_label = english_label
            current_lines = []
        elif current_label is not None and stripped:
            # Skip obvious UI noise lines
            if not _is_noise_line(stripped):
                current_lines.append(stripped)

    # Don't forget last section
    if not stop and current_label is not None and current_lines:
        result_sections.append((current_label, current_lines))

    if not result_sections:
        # No section headers found — return the raw text stripped of obvious noise
        return _strip_noise(raw_text)

    # Format into clean output
    output_parts: list[str] = []
    for label, section_lines in result_sections:
        output_parts.append(label)
        output_parts.extend(section_lines)
        output_parts.append("")

    return _clean_ws("\n".join(output_parts))


def _is_noise_line(line: str) -> bool:
    """Return True for lines that are clearly UI chrome, not JD content."""
    noise_patterns = [
        r"^(требуемый уровень кандидата|required candidate level|candidate level)$",
        r"^\d+\s*(просмотр|подписчик|вакансии|история)",   # view/follower counts
        r"^(ереван|yerevan|armenia)$",
        r"^(полная ставка|full time|part time|remote)$",
        r"^(постоянный|permanent|contract|контракт)$",
        r"^(старший|middle|junior|senior|intern)$",
        r"^(программирование|software development)$",
        r"^https?://",
        r"^\d{1,2}\s+(июня|июля|августа|января|февраля|марта|апреля|мая|сентября|октября|ноября|декабря)",
        r"^(поделитесь|share this|подать заявку|apply|откликнуться)",
        r"^fb-icon|job-poster|few-more-jobs",
        r"^\[",    # markdown links like [Программирование](...)
        r"^!?\[",  # markdown images
    ]
    low = line.lower().strip()
    for pattern in noise_patterns:
        if re.match(pattern, low):
            return True
    # Very short lines that are just numbers or single words (icon alt text etc.)
    if len(line.strip()) <= 3:
        return True
    return False


def _strip_noise(text: str) -> str:
    """Remove obvious UI noise from a raw paste when no section headers found."""
    lines = text.splitlines()
    clean = [l for l in lines if l.strip() and not _is_noise_line(l.strip())]
    return _clean_ws("\n".join(clean))


# ── per-site scrapers ─────────────────────────────────────────────────────────

def _scrape_linkedin(url: str) -> str:
    job_id = None
    for pattern in [r"currentJobId=(\d+)", r"/jobs/view/(\d+)"]:
        m = re.search(pattern, url)
        if m:
            job_id = m.group(1)
            break

    if job_id:
        api_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
        try:
            r = requests.get(api_url, headers=_HEADERS, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                jd = soup.find("div", {"class": "show-more-less-html__markup"})
                if jd:
                    text = _html_to_text(str(jd))
                    if text and not _is_linkedin_wall(text):
                        return text
        except Exception:
            pass

    r, soup = _get_soup(url)
    if soup:
        jd = soup.find("div", {"class": "show-more-less-html__markup"})
        if jd:
            text = _html_to_text(str(jd))
            if not _is_linkedin_wall(text):
                return text

    return (
        "LinkedIn requires you to be logged in to view job descriptions.\n\n"
        "Please:\n"
        "1. Open the LinkedIn job posting in your browser.\n"
        "2. Copy the full job description text.\n"
        "3. Paste it into the 'Job Description' text box on the left."
    )


_STAFF_AM_PASTE_HINT = """staff.am loads its job content via JavaScript, so it cannot be scraped automatically.

Here is what to do:
1. Open the job URL in your browser.
2. Select all text on the page (Ctrl+A / Cmd+A).
3. Copy it (Ctrl+C / Cmd+C).
4. Use the "Paste from staff.am page" expander below the text box and paste there.

The tool will automatically extract just the Обязанности / Требования / Skills sections."""


def _scrape_staff_am(url: str) -> str:
    r, soup = _get_soup(url)
    if soup is None:
        return _STAFF_AM_PASTE_HINT

    # Try __NEXT_DATA__ — works if staff.am SSR is enabled for this route
    next_data = _extract_next_data(soup)
    if next_data:
        jd_text = _staff_am_from_next_data(next_data)
        if jd_text:
            return jd_text

    # Fall back to og:description preview + paste hint
    og = _meta_og_description(soup)
    preview = (og + "\n\n") if og else ""

    return preview + _STAFF_AM_PASTE_HINT


def _scrape_indeed(url: str) -> str:
    r, soup = _get_soup(url)
    if soup:
        jd = soup.find("div", {"id": "jobDescriptionText"})
        if jd:
            return _html_to_text(str(jd))
    return ""


def _scrape_generic(url: str) -> str:
    r, soup = _get_soup(url)
    if soup is None:
        return "Could not reach the page. Please check the URL or paste the JD manually."

    next_data = _extract_next_data(soup)
    if next_data:
        found = _walk_for_value(
            next_data.get("props", {}).get("pageProps", {}), _JD_KEYS
        )
        if found:
            return _html_to_text(found)

    for tag, attrs in [
        ("div",     {"class": "job-description"}),
        ("div",     {"class": "jobDescription"}),
        ("div",     {"id":    "job-description"}),
        ("div",     {"class": "description"}),
        ("section", {"class": "description"}),
        ("article", {}),
    ]:
        el = soup.find(tag, attrs) if attrs else soup.find(tag)
        if el:
            text = _html_to_text(str(el))
            if len(text) > 100:
                return text

    paras = [_html_to_text(str(p)) for p in soup.find_all("p") if len(p.get_text()) > 30]
    if paras:
        return _clean_ws("\n".join(paras))

    og = _meta_og_description(soup)
    if og:
        return og + "\n\n[Note: Only a preview was available. Please paste the full JD.]"

    return "Could not extract a job description from this page. Please paste it manually."


# ── public entry point ────────────────────────────────────────────────────────

def scrape_job_description(url: str) -> str:
    host = _hostname(url)
    if "linkedin.com" in host:
        return _scrape_linkedin(url)
    if "staff.am" in host:
        return _scrape_staff_am(url)
    if "indeed.com" in host:
        result = _scrape_indeed(url)
        return result if result else _scrape_generic(url)
    return _scrape_generic(url)
