"""
Cone and box searches on VizieR that keep working when one server is down.

CDS in Strasbourg runs VizieR and most users only ever reach that server. When
its database is unreachable it still answers with HTTP 200, but the VOTable
holds nothing except an error message. astroquery reads that the same way as
an empty field and caches it for a week, so a single bad answer used to look
like "no stars here" for days afterwards. Here each answer is checked for that
error message, nothing is read from or written to the cache, and the official
mirrors are tried in turn until one answers.
"""

import re

from astroquery.exceptions import TableParseError
from astroquery.vizier import Vizier
from requests import exceptions as request_exceptions

from ._log import make_logger

# CDS first, then the official mirrors. CfA is left out because it passes every
# query back to CDS, so it fails whenever CDS does.
VIZIER_SERVERS = (
    "vizier.cds.unistra.fr",
    "vizier.idia.ac.za",
    "vizier.nao.ac.jp",
    "vizier.iucaa.in",
    "vizier.china-vo.org",
    "vizier.inasan.ru",
)

# Seconds to wait for a connection, then for each piece of the answer. During
# the October 2026 CDS outage queries hung for up to a minute before failing.
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60

_ERROR_INFO = re.compile(rb'<INFO[^>]*\bname="Error"[^>]*>', re.IGNORECASE)
_VALUE = re.compile(rb'\bvalue="([^"]*)"')


class VizierUnavailable(OSError):
    """
    No VizieR server gave a usable answer.

    failures holds (server, reason) for each one tried, and timed_out is True
    when every one of them failed by timing out rather than refusing or
    sending back an error.
    """

    def __init__(self, catalog, failures):
        self.catalog = catalog
        self.failures = list(failures)
        self.timed_out = bool(self.failures) and all(f[1].startswith("timed out") for f in self.failures)
        tried = "; ".join(f"{server}: {reason}" for server, reason in self.failures)
        super().__init__(f"VizieR couldn't be reached for {catalog} ({tried}). The service has "
                         "had outages lately, so please try again later.")


def _error_message(content):
    """The text of VizieR's error message if the answer is one, otherwise None."""
    match = _ERROR_INFO.search(content)
    if match is None:
        return None
    value = _VALUE.search(match.group(0))
    if value and value.group(1).strip():
        return value.group(1).decode("utf-8", "replace").strip()
    end = content.find(b"</INFO>", match.end())
    text = content[match.end():end].decode("utf-8", "replace").strip() if end != -1 else ""
    return text or "VizieR sent back an error"


def _reason(error):
    """A short description of why a request failed, for the log."""
    text = str(error)
    if isinstance(error, request_exceptions.Timeout) or "timed out" in text.lower():
        return "timed out"
    if isinstance(error, request_exceptions.ConnectionError):
        return "couldn't connect"
    if isinstance(error, request_exceptions.HTTPError) and error.response is not None:
        return f"HTTP {error.response.status_code}"
    return text.splitlines()[0][:120] if text else type(error).__name__


class VizierServers:
    """
    Which VizieR servers to try, and in what order, across several queries.

    A server that answers goes to the front for the next query. One that
    times out or refuses the connection is left out from then on, so later
    queries don't wait on it again. A server that answers with an error
    about one query (VizieR's "Request has been interrupted", say) is kept,
    since the next query may well work there.
    """

    def __init__(self, servers=VIZIER_SERVERS):
        self._servers = list(servers)
        self.unreachable = set()
        self.answered = None

    def order(self):
        """The servers to try next, best first."""
        usable = [s for s in self._servers if s not in self.unreachable]
        if self.answered in usable:
            usable.remove(self.answered)
            usable.insert(0, self.answered)
        return usable

    def record(self, server, reason=None):
        """Note how a server did: reason is None when it answered properly."""
        if reason is None:
            self.answered = server
        elif reason in ("timed out", "couldn't connect"):
            self.unreachable.add(server)
            if self.answered == server:
                self.answered = None

    @property
    def all_unreachable(self):
        return not self.order()


def query_region_tables(catalog, coordinates, *, columns, column_filters=None, radius=None, width=None,
                        servers=VIZIER_SERVERS, write_callback=None, cancel_event=None):
    """
    Search a VizieR catalogue around a position and return every table that
    has rows there, as a dictionary of astropy Tables keyed by table name.

    catalog can be one table ("II/246/out") or a whole catalogue ("IX/69"),
    which may hold several tables. Pass either radius (a cone) or width (a
    square box). An empty dictionary means the server answered properly but
    found nothing. Raises VizierUnavailable if every server failed, and
    returns None if cancel_event gets set.

    servers is a list of server names, or a VizierServers to share what's
    been learned about the servers with other queries.
    """
    log = make_logger(write_callback)
    memory = servers if isinstance(servers, VizierServers) else None
    servers = memory.order() if memory is not None else list(servers)
    failures = []
    for number, server in enumerate(servers):
        if cancel_event is not None and cancel_event.is_set():
            return None
        vizier = Vizier(columns=columns, row_limit=-1, column_filters=column_filters or {})
        vizier.VIZIER_SERVER = server
        vizier.TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)
        try:
            response = vizier.query_region_async(coordinates, radius=radius, width=width,
                                                 catalog=catalog, cache=False)
            message = _error_message(response.content)
            if message is not None:
                raise OSError(f"VizieR error: {message}")
            response.raise_for_status()
            tables = vizier._parse_result(response)
            if tables is None or isinstance(tables, str):
                raise OSError("the answer wasn't a VOTable")
        except (request_exceptions.RequestException, OSError, TableParseError) as error:
            failures.append((server, _reason(error)))
            if memory is not None:
                memory.record(server, failures[-1][1])
            if number + 1 < len(servers):
                log(f"VizieR at {server} didn't answer properly ({failures[-1][1]}); "
                    f"trying {servers[number + 1]}.")
            continue
        if memory is not None:
            memory.record(server)
        if number > 0:
            log(f"Got the {catalog} results from {server}.")
        return {name: tables[name] for name in tables.keys()}
    raise VizierUnavailable(catalog, failures)


def query_region(catalog, coordinates, *, columns, column_filters=None, radius=None, width=None,
                 servers=VIZIER_SERVERS, write_callback=None, cancel_event=None):
    """
    Search one VizieR catalogue around a position.

    Pass either radius (a cone) or width (a square box). Returns the
    catalogue's astropy Table, or None when the server answered properly but
    found nothing in the area. Raises VizierUnavailable if every server
    failed, and returns None straight away if cancel_event gets set.
    """
    tables = query_region_tables(catalog, coordinates, columns=columns, column_filters=column_filters,
                                 radius=radius, width=width, servers=servers,
                                 write_callback=write_callback, cancel_event=cancel_event)
    if not tables:
        return None
    # An empty field comes back with no table at all, which isn't an error
    if catalog in tables:
        return tables[catalog]
    return next(iter(tables.values())) if len(tables) == 1 else None
