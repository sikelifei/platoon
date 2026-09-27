import gzip
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from xml.etree import ElementTree

import pytest

from platoon.fanoutqa.inference_scripts.kiwix_compat import ATOM_NAMESPACE, _proxy_handler

SEARCH_HTML = """<!doctype html>
<html><body>
<div class="header"><a href="/wiki/A/Header">Header navigation</a></div>
<div class="results"><ul>
<li>
  <a href="/wiki/A/Caf%C3%A9?x=1&amp;y=2">Café &amp; Tea</a>
  <cite><a href="/wiki/A/Snippet">snippet link</a></cite>
</li>
<li>
  <span><a href="/wiki/A/Nested">nested link is not a title</a></span>
  <a href="/wiki/A/Second">Second &lt;Title&gt;</a>
</li>
<li><a href="/wiki/A/Second">Second &lt;Title&gt;</a></li>
</ul></div>
<div class="footer"><a href="/search?start=2">Next page</a></div>
</body></html>""".encode("utf-8")

ZERO_RESULTS_HTML = b'<html><body><div class="results"><ul></ul></div></body></html>'


@pytest.fixture
def proxy_stack():
    observed_paths = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            observed_paths.append(self.path)
            path = urllib.parse.urlsplit(self.path).path
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            pattern = query.get("pattern", [""])[0]

            status = 200
            content_type = "text/html; charset=utf-8"
            content_encoding = None
            if path == "/search" and pattern == "zero":
                body = ZERO_RESULTS_HTML
            elif path == "/search" and pattern == "empty":
                body = b""
            elif path == "/search" and pattern == "gzip":
                body = gzip.compress(SEARCH_HTML)
                content_encoding = "gzip"
            elif path == "/search" and pattern == "invalid-html":
                body = b"<html><body>upstream error page without result markup</body></html>"
            elif path == "/search" and pattern == "invalid-gzip":
                body = gzip.compress(b"<html><body>compressed error page without results</body></html>")
                content_encoding = "gzip"
            elif path == "/search" and pattern == "not-html":
                content_type = "application/json"
                body = b'{"results":[]}'
            elif path == "/search" and pattern == "missing":
                status = 404
                body = b"<html><body>not found</body></html>"
            elif path == "/raw":
                content_type = "application/octet-stream"
                body = b"\x00raw response\xff"
            else:
                body = SEARCH_HTML

            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if content_encoding:
                self.send_header("Content-Encoding", content_encoding)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, format, *args):
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream.daemon_threads = True
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    upstream_base = f"http://127.0.0.1:{upstream.server_address[1]}"
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), _proxy_handler(upstream_base))
    proxy.daemon_threads = True
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    proxy_base = f"http://127.0.0.1:{proxy.server_address[1]}"

    try:
        yield proxy_base, observed_paths
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=2)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.headers, response.read()


def test_search_converts_only_ranked_title_links_and_preserves_raw_query(proxy_stack):
    proxy_base, observed_paths = proxy_stack
    raw_query = "pattern=Paris%20%26%20Rome&start=000&pageLength=3&books.name=wiki%2Fsample"

    status, headers, body = _get(f"{proxy_base}/search?{raw_query}")

    assert observed_paths == [f"/search?{raw_query}"]
    assert status == 200
    assert headers.get("Content-Type") == "application/atom+xml; charset=utf-8"
    assert len(headers.get_all("Content-Length")) == 1
    assert int(headers.get("Content-Length")) == len(body)
    assert headers.get("Content-Encoding") is None
    assert b"Caf\xc3\xa9 &amp; Tea" in body
    assert b"/wiki/A/Caf%C3%A9?x=1&amp;y=2" in body

    feed = ElementTree.fromstring(body)
    entries = feed.findall(f"{{{ATOM_NAMESPACE}}}entry")
    assert [entry.findtext(f"{{{ATOM_NAMESPACE}}}title") for entry in entries] == [
        "Café & Tea",
        "Second <Title>",
        "Second <Title>",
    ]
    assert [entry.find(f"{{{ATOM_NAMESPACE}}}link").get("href") for entry in entries] == [
        "/wiki/A/Caf%C3%A9?x=1&y=2",
        "/wiki/A/Second",
        "/wiki/A/Second",
    ]


def test_search_with_no_results_returns_an_empty_atom_feed(proxy_stack):
    proxy_base, _ = proxy_stack

    status, headers, body = _get(f"{proxy_base}/search?pattern=zero&pageLength=3")

    assert status == 200
    assert headers.get("Content-Type") == "application/atom+xml; charset=utf-8"
    feed = ElementTree.fromstring(body)
    assert feed.findall(f"{{{ATOM_NAMESPACE}}}entry") == []


@pytest.mark.parametrize("pattern", ["invalid-html", "invalid-gzip"])
def test_html_error_without_results_container_is_not_misreported_as_empty(pattern, proxy_stack):
    proxy_base, _ = proxy_stack

    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(f"{proxy_base}/search?pattern={pattern}", timeout=5)

    response = caught.value
    assert response.code == 502
    assert response.headers.get("Content-Type") == "text/plain; charset=utf-8"
    body = response.read()
    assert b"did not contain a .results container" in body
    assert len(response.headers.get_all("Content-Length")) == 1
    assert int(response.headers.get("Content-Length")) == len(body)
    assert response.headers.get("Content-Encoding") is None


def test_empty_search_body_and_non_html_search_body_pass_through(proxy_stack):
    proxy_base, _ = proxy_stack

    empty_status, empty_headers, empty_body = _get(f"{proxy_base}/search?pattern=empty")
    json_status, json_headers, json_body = _get(f"{proxy_base}/search?pattern=not-html")

    assert empty_status == 200
    assert empty_headers.get("Content-Type") == "text/html; charset=utf-8"
    assert empty_body == b""
    assert json_status == 200
    assert json_headers.get("Content-Type") == "application/json"
    assert json_body == b'{"results":[]}'


def test_non_search_body_status_and_content_type_pass_through(proxy_stack):
    proxy_base, observed_paths = proxy_stack

    status, headers, body = _get(f"{proxy_base}/raw?keep=%2F&x=1")

    assert status == 200
    assert headers.get("Content-Type") == "application/octet-stream"
    assert len(headers.get_all("Content-Length")) == 1
    assert body == b"\x00raw response\xff"
    assert observed_paths == ["/raw?keep=%2F&x=1"]


def test_upstream_non_success_status_and_body_pass_through(proxy_stack):
    proxy_base, _ = proxy_stack

    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(f"{proxy_base}/search?pattern=missing", timeout=5)

    response = caught.value
    assert response.code == 404
    assert response.headers.get("Content-Type") == "text/html; charset=utf-8"
    assert response.read() == b"<html><body>not found</body></html>"


def test_gzip_search_is_decoded_then_transformed_without_encoding_header(proxy_stack):
    proxy_base, _ = proxy_stack

    status, headers, body = _get(f"{proxy_base}/search?pattern=gzip&pageLength=3")

    assert status == 200
    assert headers.get("Content-Type") == "application/atom+xml; charset=utf-8"
    assert headers.get("Content-Encoding") is None
    assert len(headers.get_all("Content-Length")) == 1
    assert int(headers.get("Content-Length")) == len(body)
    feed = ElementTree.fromstring(body)
    entries = feed.findall(f"{{{ATOM_NAMESPACE}}}entry")
    assert [entry.findtext(f"{{{ATOM_NAMESPACE}}}title") for entry in entries] == [
        "Café & Tea",
        "Second <Title>",
        "Second <Title>",
    ]
