import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USER_AGENT = "rag-llm-chemistry-learning-project/0.1"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class ChemistryApiError(RuntimeError):
    """Controlled failure while querying a chemistry source."""


def get_json(
    url: str,
    params: dict[str, str | int] | None = None,
    timeout_seconds: int = 30,
    max_attempts: int = 3,
) -> Any:
    if params:
        url = f"{url}?{urlencode(params)}"

    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )

    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = json.load(response)

            if not isinstance(payload, (dict, list)):
                raise ChemistryApiError(
                    f"The API returned an unexpected format: {url}"
                )

            return payload
        except HTTPError as error:
            response_text = error.read(500).decode(
                "utf-8",
                errors="replace",
            )

            if (
                error.code in RETRYABLE_STATUS_CODES
                and attempt < max_attempts
            ):
                retry_after = error.headers.get("Retry-After", "")

                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = float(attempt)

                time.sleep(min(delay, 10.0))
                continue

            raise ChemistryApiError(
                f"HTTP {error.code} while querying {url}: "
                f"{response_text}"
            ) from error
        except (TimeoutError, URLError) as error:
            if attempt < max_attempts:
                time.sleep(float(attempt))
                continue

            raise ChemistryApiError(
                f"Could not connect to {url}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise ChemistryApiError(
                f"The API did not return valid JSON: {url}"
            ) from error

    raise ChemistryApiError(f"The request could not be completed: {url}")


def get_text(
    url: str,
    params: dict[str, str | int] | None = None,
    timeout_seconds: int = 30,
    max_attempts: int = 3,
) -> str:
    if params:
        url = f"{url}?{urlencode(params)}"

    request = Request(
        url,
        headers={
            "Accept": "text/html, text/plain;q=0.9, */*;q=0.8",
            "User-Agent": USER_AGENT,
        },
    )

    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            if (
                error.code in RETRYABLE_STATUS_CODES
                and attempt < max_attempts
            ):
                time.sleep(float(attempt))
                continue

            raise ChemistryApiError(
                f"HTTP {error.code} while querying {url}."
            ) from error
        except (TimeoutError, URLError) as error:
            if attempt < max_attempts:
                time.sleep(float(attempt))
                continue

            raise ChemistryApiError(
                f"Could not connect to {url}: {error}"
            ) from error

    raise ChemistryApiError(f"The request could not be completed: {url}")
