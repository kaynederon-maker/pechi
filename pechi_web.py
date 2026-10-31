"""Web tools for Pechi: search with ddgs, read and scrape with httpx and BeautifulSoup.

Pages are untrusted. Only public addresses are reachable, every redirect is checked
again, and a response is cut off at MAX_BYTES.
"""

import ipaddress
import socket
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

MAX_BYTES = 2_000_000
MAX_REDIRECTS = 5
MAX_MATCHES = 40
MAX_LINKS = 15
MAX_SEARCH_RESULTS = 8
MAX_PAGE_CHARS = 4500  # leaves room for the links inside one observation
TIMEOUT = 20
HEADERS = {"User-Agent": "Pechi/1.0 (https://github.com/kaynederon-maker/pechi)"}
TEXT_TYPES = ("text/", "application/json", "application/xhtml", "application/xml")
NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "footer",
    "aside",
    "form",
    "iframe",
)


class Page(NamedTuple):
    url: str
    text: str
    is_html: bool


def require_public(url: str) -> None:
    """Refuse other schemes and private hosts, so a page cannot aim Pechi at your LAN."""
    # ponytail: DNS could change between this check and the request. Pin the resolved
    # address if that ever matters.
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise PermissionError(f"{url} is not an http or https address")
    try:
        found = socket.getaddrinfo(
            parts.hostname, parts.port or 443, type=socket.SOCK_STREAM
        )
    except socket.gaierror as err:
        raise ValueError(f"cannot find {parts.hostname}") from err
    for *_, sockaddr in found:
        address = ipaddress.ip_address(sockaddr[0].split("%")[0])
        address = getattr(address, "ipv4_mapped", None) or address
        if not address.is_global or address.is_multicast:
            raise PermissionError(f"{parts.hostname} is a private address")


def same_site(first: str, second: str) -> bool:
    hosts = [
        (urlsplit(url).hostname or "").removeprefix("www.") for url in (first, second)
    ]
    return hosts[0] == hosts[1]


def read_body(response: httpx.Response) -> bytes:
    body = bytearray()
    for chunk in response.iter_bytes():
        body += chunk
        if len(body) >= MAX_BYTES:
            break
    return bytes(body[:MAX_BYTES])


def fetch(url: str, transport: httpx.BaseTransport | None = None) -> Page:
    """GET a text page, following same-site redirects by hand so each hop is checked."""
    with httpx.Client(headers=HEADERS, timeout=TIMEOUT, transport=transport) as client:
        for _ in range(MAX_REDIRECTS + 1):
            require_public(url)
            with client.stream("GET", url) as response:
                if response.is_redirect:
                    target = urljoin(url, response.headers["location"])
                    if not same_site(url, target):  # approval covered one site only
                        raise ValueError(
                            f"this page redirects to another site: {target}. "
                            "Fetch that address if you still want it."
                        )
                    url = target
                    continue
                response.raise_for_status()
                kind = response.headers.get("content-type", "text/html").lower()
                if not kind.startswith(TEXT_TYPES):
                    raise ValueError(f"{kind} is not a text page")
                text = read_body(response).decode(
                    response.encoding or "utf-8", errors="replace"
                )
                return Page(str(response.url), text, "html" in kind)
    raise ValueError(f"more than {MAX_REDIRECTS} redirects")


def page_links(soup: BeautifulSoup, base: str) -> list[str]:
    found = {}
    for anchor in soup.select("a[href]"):
        target = urljoin(base, anchor["href"]).split("#")[0]
        label = anchor.get_text(" ", strip=True)
        if target.startswith("http") and label:
            found.setdefault(target, label[:60])
    return [f"{label}: {url}" for url, label in list(found.items())[:MAX_LINKS]]


def readable(page: Page) -> str:
    """Title, main text and links, with page furniture removed."""
    if not page.is_html:
        return f"URL: {page.url}\n\n{page.text[:MAX_PAGE_CHARS]}"
    soup = BeautifulSoup(page.text, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else "(no title)"
    links = page_links(soup, page.url)
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    main = (
        soup.select_one("#mw-content-text")  # Wikipedia keeps its menus inside main
        or soup.select_one("main, article, [role=main]")
        or soup
    )
    lines = (line.strip() for line in main.get_text("\n").splitlines())
    body = "\n".join(line for line in lines if line)[:MAX_PAGE_CHARS]
    return f"Title: {title}\nURL: {page.url}\n\n{body}\n\nLinks:\n" + "\n".join(links)


def scrape(page: Page, selector: str) -> str:
    """Text of every element matching a CSS selector, with its link when it has one."""
    soup = BeautifulSoup(page.text, "lxml")
    found = soup.select(selector, limit=MAX_MATCHES)
    if not found:
        return f"No elements match {selector} on {page.url}."
    rows = []
    for tag in found:
        text = tag.get_text(" ", strip=True)
        link = tag.get("href") or (tag.select_one("a[href]") or {}).get("href")
        rows.append(f"{text} -> {urljoin(page.url, link)}" if link else text)
    return f"{len(rows)} matches on {page.url}\n" + "\n".join(rows)


def web_search(root: Path, query: str, max_results: int = 5) -> str:
    from ddgs import DDGS  # loaded here because importing it takes about a second

    limit = min(int(max_results), MAX_SEARCH_RESULTS)
    hits = DDGS().text(query, max_results=limit)
    rows = [f"{hit['title']}\n{hit['href']}\n{hit['body']}" for hit in hits]
    return "\n\n".join(rows) or "No results."


def web_fetch(root: Path, url: str, selector: str = "") -> str:
    # ponytail: no JavaScript, so pages that render client-side come back thin.
    # Add a headless browser here if that becomes the common case.
    page = fetch(url)
    return scrape(page, selector) if selector else readable(page)
