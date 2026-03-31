import base64
import logging
import os
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading

from dotenv import load_dotenv
from clickhouse_connect import get_client

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD")
CLICKHOUSE_TABLE = os.getenv("CLICKHOUSE_TABLE")

if not all([CLICKHOUSE_HOST, CLICKHOUSE_DATABASE, CLICKHOUSE_PASSWORD, CLICKHOUSE_TABLE]):
    raise ValueError("Missing ClickHouse env vars: CLICKHOUSE_HOST, CLICKHOUSE_DATABASE, CLICKHOUSE_PASSWORD, CLICKHOUSE_TABLE")

LEAF_API_URL = os.getenv("LEAF_API_URL")
LEAF_API_KEY = os.getenv("LEAF_API_KEY")
LEAF_API_SECRET = os.getenv("LEAF_API_SECRET")
LEAF_API_PORT = os.getenv("LEAF_API_PORT")
LEAF_API_TIMEOUT = float(os.getenv("LEAF_API_TIMEOUT", "30"))
LEAF_MAX_WORKERS = int(os.getenv("LEAF_MAX_WORKERS", "8"))
LEAF_AUTH_SCHEME = os.getenv("LEAF_AUTH_SCHEME", "auto").lower()
LEAF_AUTH_HEADER = os.getenv("LEAF_AUTH_HEADER")
LEAF_AUTH_TOKEN = os.getenv("LEAF_AUTH_TOKEN")
LEAF_SEND_X_HEADERS = os.getenv("LEAF_SEND_X_HEADERS", "true").lower() in ("1", "true", "yes", "on")
LEAF_TOKEN_ENDPOINT = os.getenv("LEAF_TOKEN_ENDPOINT", "/api/token")
LEAF_TOKEN_METHOD = os.getenv("LEAF_TOKEN_METHOD", "POST").upper()
LEAF_TOKEN_KEY_FIELD = os.getenv("LEAF_TOKEN_KEY_FIELD", "client_id")
LEAF_TOKEN_SECRET_FIELD = os.getenv("LEAF_TOKEN_SECRET_FIELD", "client_secret")
LEAF_TOKEN_FIELD = os.getenv("LEAF_TOKEN_FIELD")
LEAF_TOKEN_TTL_SECONDS = int(os.getenv("LEAF_TOKEN_TTL_SECONDS", "300"))
LEAF_TOKEN_CLIENT = os.getenv("LEAF_TOKEN_CLIENT")
LEAF_TOKEN_SECRET = os.getenv("LEAF_TOKEN_SECRET")

if not LEAF_API_URL:
    raise ValueError("Missing API env var: LEAF_API_URL")
if not all([LEAF_TOKEN_CLIENT, LEAF_TOKEN_SECRET]):
    raise ValueError("Missing token env vars: LEAF_TOKEN_CLIENT, LEAF_TOKEN_SECRET")

logger.info(
    "Auth config: scheme=%s, send_x_headers=%s, custom_auth_header=%s",
    LEAF_AUTH_SCHEME,
    LEAF_SEND_X_HEADERS,
    bool(LEAF_AUTH_HEADER),
)

client = get_client(
    host=CLICKHOUSE_HOST,
    username="default",
    password=CLICKHOUSE_PASSWORD,
    database=CLICKHOUSE_DATABASE,
)

STATE_FILE = "ch_cursor.txt"

def format_cursor(value):
    return value.strftime("%Y-%m-%d %H:%M:%S")

def parse_cursor(value):
    if isinstance(value, datetime):
        dt_value = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            dt_value = datetime.fromisoformat(text)
        except ValueError:
            dt_value = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")

    if dt_value.tzinfo is not None:
        dt_value = dt_value.astimezone(timezone.utc).replace(tzinfo=None)
    return dt_value

def yesterday_start():
    now = datetime.now()
    return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

def load_cursor():
    try:
        with open(STATE_FILE, "r") as f:
            value = f.read().strip()
            if value:
                return parse_cursor(value)
    except FileNotFoundError:
        pass
    return yesterday_start()

def save_cursor(cursor):
    with open(STATE_FILE, "w") as f:
        f.write(format_cursor(cursor))

COL_OPERATION_NAME = 0
COL_CONTENTS_ID = 1
COL_CONTENTS_NAME = 2
COL_SCHOOL_ID = 3
COL_CONTEXT_LABEL = 4
COL_TIMESTAMP = 5

TOKEN_LOCK = threading.Lock()
TOKEN_VALUE = None
TOKEN_EXPIRES_AT = None
TOKEN_TYPE = None

def build_auth_headers():
    headers = {}
    if LEAF_SEND_X_HEADERS:
        if LEAF_API_KEY:
            headers["X-API-KEY"] = LEAF_API_KEY
        else:
            logger.warning("LEAF_SEND_X_HEADERS enabled but LEAF_API_KEY is not set")
        if LEAF_API_SECRET:
            headers["X-API-SECRET"] = LEAF_API_SECRET
        else:
            logger.warning("LEAF_SEND_X_HEADERS enabled but LEAF_API_SECRET is not set")

    if LEAF_AUTH_HEADER:
        headers["Authorization"] = LEAF_AUTH_HEADER
        return headers

    if LEAF_AUTH_SCHEME == "none":
        return headers
    if LEAF_AUTH_SCHEME == "basic":
        token = base64.b64encode(f"{LEAF_API_KEY}:{LEAF_API_SECRET}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
        return headers
    if LEAF_AUTH_SCHEME == "basic-token":
        if not LEAF_AUTH_TOKEN:
            raise ValueError("LEAF_AUTH_TOKEN is required for LEAF_AUTH_SCHEME=basic-token")
        headers["Authorization"] = f"Basic {LEAF_AUTH_TOKEN}"
        return headers
    if LEAF_AUTH_SCHEME == "bearer":
        token = LEAF_AUTH_TOKEN or get_auth_token()
        headers["Authorization"] = f"Bearer {token}"
        return headers
    if LEAF_AUTH_SCHEME == "token":
        token = LEAF_AUTH_TOKEN or get_auth_token()
        headers["Authorization"] = f"Token {token}"
        return headers
    if LEAF_AUTH_SCHEME == "apikey":
        token = LEAF_AUTH_TOKEN or get_auth_token()
        headers["Authorization"] = f"ApiKey {token}"
        return headers
    if LEAF_AUTH_SCHEME == "auto":
        token = LEAF_AUTH_TOKEN or get_auth_token()
        scheme = TOKEN_TYPE or "Token"
        headers["Authorization"] = f"{scheme} {token}"
        return headers

    raise ValueError(f"Unsupported LEAF_AUTH_SCHEME: {LEAF_AUTH_SCHEME}")

def build_base_url():
    base_url = LEAF_API_URL.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        logger.warning("LEAF_API_URL missing scheme, defaulting to http://")
        base_url = f"http://{base_url}"
    if LEAF_API_PORT:
        split = urllib.parse.urlsplit(base_url)
        netloc = split.netloc
        if ":" not in netloc:
            netloc = f"{netloc}:{LEAF_API_PORT}"
        base_url = urllib.parse.urlunsplit((split.scheme, netloc, split.path.rstrip("/"), "", ""))
    return base_url

def build_pdf_url(content_id):
    base_url = build_base_url()
    endpoint = "/api/get_pdf_by_id"
    query = urllib.parse.urlencode({"content_id": content_id})
    return f"{base_url}{endpoint}?{query}"

def build_token_url():
    base_url = build_base_url()
    endpoint = LEAF_TOKEN_ENDPOINT.strip()
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return f"{base_url}{endpoint}"

def get_auth_token():
    global TOKEN_VALUE, TOKEN_EXPIRES_AT, TOKEN_TYPE
    now = datetime.now()
    with TOKEN_LOCK:
        if TOKEN_VALUE and TOKEN_EXPIRES_AT and now < TOKEN_EXPIRES_AT:
            return TOKEN_VALUE

        url = build_token_url()
        payload = {
            LEAF_TOKEN_KEY_FIELD: LEAF_TOKEN_CLIENT,
            LEAF_TOKEN_SECRET_FIELD: LEAF_TOKEN_SECRET,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        logger.info("Requesting auth token from %s with fields %s", url, list(payload.keys()))
        request = urllib.request.Request(url, data=data, headers=headers, method=LEAF_TOKEN_METHOD)
        try:
            with urllib.request.urlopen(request, timeout=LEAF_API_TIMEOUT) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = "<unreadable response body>"
            raise RuntimeError(f"Token request HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Failed to reach token endpoint") from exc

        token, expires_at, token_type = parse_token_response(body, now)
        TOKEN_VALUE = token
        TOKEN_EXPIRES_AT = expires_at
        TOKEN_TYPE = token_type
        if TOKEN_TYPE:
            logger.info("Token type from response: %s", TOKEN_TYPE)
        return TOKEN_VALUE

def parse_token_response(body, now):
    token = None
    token_type = None
    expires_at = now + timedelta(seconds=LEAF_TOKEN_TTL_SECONDS)
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        if LEAF_TOKEN_FIELD:
            token = data.get(LEAF_TOKEN_FIELD)
        if not token:
            for key in ("access_token", "token", "jwt", "auth_token"):
                if key in data:
                    token = data[key]
                    break
        if "token_type" in data:
            token_type = data["token_type"]
        if "expires_in" in data:
            try:
                expires_at = now + timedelta(seconds=int(data["expires_in"]))
            except (TypeError, ValueError):
                pass
        if "expires_at" in data:
            try:
                expires_at = parse_cursor(data["expires_at"])
            except ValueError:
                pass

    if not token:
        token = body.strip()

    if not token:
        raise RuntimeError("Token response did not include a token")
    return token, expires_at, token_type

def check_pdf_endpoint(content_id):
    url = build_pdf_url(content_id)
    logger.info("Checking PDF URL: %s", url)
    request = urllib.request.Request(url, headers=build_auth_headers())
    try:
        with urllib.request.urlopen(request, timeout=LEAF_API_TIMEOUT) as response:
            if response.status == 200:
                logger.info("PDF URL OK: %s", url)
                return
            raise RuntimeError(f"Unexpected status {response.status} for {content_id}")
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = "<unreadable response body>"
        raise RuntimeError(f"HTTP {exc.code} while fetching {content_id}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach API for {content_id}") from exc

def handle_rows(rows):

    # rows is a list of tuples/dicts depending on query format
    logger.info("Processing %s rows", len(rows))
    content_ids = []
    for row in rows:
        if row[COL_OPERATION_NAME] != "REGISTER_CONTENTS":
            continue
        content_id = row[COL_CONTENTS_ID]
        if not content_id:
            logger.info("Skipping row with empty contents_id")
            continue
        content_ids.append(content_id)

    if not content_ids:
        return

    logger.info("Checking %s content IDs with %s workers", len(content_ids), LEAF_MAX_WORKERS)

    def worker(content_id):
        try:
            check_pdf_endpoint(content_id)
            return None
        except RuntimeError as exc:
            return f"Failed to check PDF URL for content_id={content_id}: {exc}"

    with ThreadPoolExecutor(max_workers=LEAF_MAX_WORKERS) as executor:
        futures = [executor.submit(worker, content_id) for content_id in content_ids]
        for future in as_completed(futures):
            error = future.result()
            if error:
                logger.error("%s", error)

def main():
    last_seen = max(load_cursor(), yesterday_start())
    logger.info("Starting cursor at %s", format_cursor(last_seen))

    while True:

        query = f"""
            SELECT operation_name, contents_id, contents_name, school_id, context_label, timestamp
            FROM {CLICKHOUSE_TABLE}
            WHERE timestamp > parseDateTimeBestEffort(%(cursor)s)
            ORDER BY timestamp
            LIMIT 5000
        """
        rows = client.query(query, parameters={"cursor": format_cursor(last_seen)}).result_rows

        if rows:
            handle_rows(rows)


            last_seen = parse_cursor(rows[-1][COL_TIMESTAMP])  # timestamp column index
            save_cursor(last_seen)
            logger.info("Advanced cursor to %s", format_cursor(last_seen))
        else:
            time.sleep(1.0)  # polling interval

if __name__ == "__main__":
    main()
