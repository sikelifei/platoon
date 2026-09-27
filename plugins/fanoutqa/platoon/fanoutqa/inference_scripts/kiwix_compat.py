"""Local stdlib proxy adapting Kiwix HTML search results to Atom for FanOutQA.

Run with ``python -m platoon.fanoutqa.inference_scripts.kiwix_compat``.
Only successful HTML responses from ``/search`` are transformed; the original
query string, article hrefs, result order, and non-search responses are kept.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import zlib
from email.message import Message
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from xml.etree import ElementTree

logger = logging.getLogger("platoon.fanoutqa.kiwix_compat")
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class _SearchResultParser(HTMLParser):
    """Read direct result-title anchors in Kiwix's results list, in document order."""

    _VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results_container_seen = False
        self.results: list[tuple[str, str]] = []
        self._results_depth = 0
        self._list_depth = 0
        self._list_item_depth = 0
        self._open_tags: list[str] = []
        self._current_href: str | None = None
        self._current_title: list[str] = []
        self._capturing_title = False

    @staticmethod
    def _is_article_href(href: str) -> bool:
        return "/A/" in urllib.parse.urlsplit(href).path

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        parent = self._open_tags[-1] if self._open_tags else None

        if tag == "div":
            classes = (attributes.get("class") or "").split()
            if self._results_depth:
                self._results_depth += 1
            elif "results" in classes:
                self.results_container_seen = True
                self._results_depth = 1
        elif self._results_depth and tag == "ul":
            self._list_depth += 1
        elif self._results_depth and self._list_depth and tag == "li":
            if self._list_item_depth == 0:
                self._current_href = None
                self._current_title = []
                self._capturing_title = False
            self._list_item_depth += 1
        elif (
            self._results_depth
            and self._list_depth == 1
            and self._list_item_depth == 1
            and parent == "li"
            and tag == "a"
            and self._current_href is None
        ):
            href = attributes.get("href")
            if href and self._is_article_href(href):
                self._current_href = href
                self._capturing_title = True

        if tag not in self._VOID_TAGS:
            self._open_tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capturing_title:
            self._capturing_title = False
        elif tag == "li" and self._results_depth and self._list_item_depth:
            if self._list_item_depth == 1:
                if self._current_href is not None:
                    title = "".join(self._current_title).strip()
                    self.results.append((title, self._current_href))
                self._current_href = None
                self._current_title = []
                self._capturing_title = False
            self._list_item_depth -= 1
        elif tag == "ul" and self._results_depth and self._list_depth:
            self._list_depth -= 1
        elif tag == "div" and self._results_depth:
            self._results_depth -= 1

        for index in range(len(self._open_tags) - 1, -1, -1):
            if self._open_tags[index] == tag:
                del self._open_tags[index:]
                break

    def handle_data(self, data: str) -> None:
        if self._capturing_title:
            self._current_title.append(data)


def _parse_kiwix_search_html(body: bytes, content_type: str) -> tuple[bytes, bool]:
    """Return Atom XML and whether a Kiwix results container was present."""
    content_headers = Message()
    content_headers["content-type"] = content_type
    charset = content_headers.get_content_charset() or "utf-8"
    try:
        text = body.decode(charset, errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")

    parser = _SearchResultParser()
    parser.feed(text)
    parser.close()
    if not parser.results_container_seen:
        return body, False

    feed = ElementTree.Element(f"{{{ATOM_NAMESPACE}}}feed")
    for title, href in parser.results:
        entry = ElementTree.SubElement(feed, f"{{{ATOM_NAMESPACE}}}entry")
        title_element = ElementTree.SubElement(entry, f"{{{ATOM_NAMESPACE}}}title")
        title_element.text = title
        link_element = ElementTree.SubElement(entry, f"{{{ATOM_NAMESPACE}}}link")
        link_element.set("href", href)
    atom = ElementTree.tostring(feed, encoding="utf-8", xml_declaration=True)
    return atom, True


def _decode_html_body(body: bytes, content_encoding: str | None) -> bytes | None:
    if not content_encoding or content_encoding.lower() == "identity":
        return body
    encodings = [part.strip().lower() for part in content_encoding.split(",")]
    decoded = body
    try:
        for encoding in reversed(encodings):
            if encoding == "gzip":
                decoded = gzip.decompress(decoded)
            elif encoding == "deflate":
                try:
                    decoded = zlib.decompress(decoded)
                except zlib.error:
                    decoded = zlib.decompress(decoded, -zlib.MAX_WBITS)
            else:
                return None
    except (OSError, zlib.error):
        return None
    return decoded


def _hop_by_hop_headers(headers) -> set[str]:
    excluded = set(_HOP_BY_HOP_HEADERS)
    for name, value in headers.items():
        if name.lower() == "connection":
            excluded.update(token.strip().lower() for token in value.split(","))
    return excluded


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _proxy_handler(upstream_base: str) -> type[BaseHTTPRequestHandler]:
    upstream = upstream_base.rstrip("/")

    class KiwixProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self) -> None:
            if not self.path.startswith("/"):
                self.send_error(400, "Expected an origin-form request path")
                return

            request_path = urllib.parse.urlsplit(self.path).path
            request_headers = {
                name: value
                for name, value in self.headers.items()
                if name.lower() not in _HOP_BY_HOP_HEADERS | {"host", "content-length"}
                and not (request_path == "/search" and name.lower() == "accept-encoding")
            }
            if request_path == "/search":
                # The Atom adapter needs readable HTML; avoid Brotli/Zstandard encodings
                # that the standard library cannot decode.
                request_headers["Accept-Encoding"] = "identity"
            request = urllib.request.Request(upstream + self.path, headers=request_headers, method="GET")
            try:
                try:
                    response = urllib.request.build_opener(_NoRedirect()).open(request, timeout=60)
                except urllib.error.HTTPError as exc:
                    response = exc
                with response:
                    status = response.getcode()
                    response_headers = response.headers
                    body = response.read()
            except urllib.error.URLError as exc:
                message = f"Upstream Kiwix request failed: {exc.reason}".encode("utf-8", errors="replace")
                self.send_response(502)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
                return

            content_type = response_headers.get("Content-Type", "")
            path = urllib.parse.urlsplit(self.path).path
            replaced_body = False
            is_html = content_type.lower().split(";", 1)[0].strip() in {"text/html", "application/xhtml+xml"}
            if path == "/search" and 200 <= status < 300 and body and is_html:
                decoded = _decode_html_body(body, response_headers.get("Content-Encoding"))
                if decoded is not None:
                    atom, has_results_container = _parse_kiwix_search_html(decoded, content_type)
                    if has_results_container:
                        body = atom
                        replaced_body = True
                        content_type = "application/atom+xml; charset=utf-8"
                    else:
                        status = 502
                        body = b"Kiwix search HTML did not contain a .results container."
                        replaced_body = True
                        content_type = "text/plain; charset=utf-8"

            self.send_response(status)
            excluded = _hop_by_hop_headers(response_headers)
            for name, value in response_headers.items():
                lowered = name.lower()
                if (
                    lowered in excluded
                    or lowered == "content-length"
                    or (replaced_body and lowered == "content-encoding")
                ):
                    continue
                if lowered == "content-type" and replaced_body:
                    self.send_header("Content-Type", content_type)
                else:
                    self.send_header(name, value)
            if replaced_body and not any(name.lower() == "content-type" for name, _ in response_headers.items()):
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

    return KiwixProxyHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adapt Kiwix HTML search responses to FanOutQA's Atom parser.")
    parser.add_argument(
        "--upstream",
        default=os.getenv("FANOUTQA_KIWIX_UPSTREAM", "http://127.0.0.1:8888"),
        help="Kiwix server base URL (default: http://127.0.0.1:8888)",
    )
    parser.add_argument(
        "--listen-host",
        default=os.getenv("FANOUTQA_KIWIX_PROXY_HOST", "127.0.0.1"),
        help="Local address to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("FANOUTQA_KIWIX_PROXY_PORT", "8889")),
        help="Local port to bind (default: 8889)",
    )
    options = parser.parse_args(argv)
    server = ThreadingHTTPServer((options.listen_host, options.port), _proxy_handler(options.upstream))
    logger.info(
        "Kiwix compatibility proxy listening on http://%s:%d; upstream %s",
        options.listen_host,
        server.server_address[1],
        options.upstream.rstrip("/"),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping Kiwix compatibility proxy")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
