from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pyarrow as pa
import pytest

from patito import sources


class MockCursor:
    """Mock database cursor keeping track of all executed database queries."""

    def __init__(self) -> None:
        """Construct new mocked cursor object that tracks executed queries."""
        self.executed_queries: List[str] = []

    def execute(self, query: str) -> None:
        # Do not query Snowflake in order to speed up test and minimize costs
        self.executed_queries.append(query)

    def fetch_arrow_all(self) -> pa.Table:
        # Return something with a correct datatype
        return pa.Table.from_pydict({"column": [1, 2, 3]})


class MockConnection:
    """Mock database connection only returning a mock cursor object."""

    def __init__(self):
        """Create new connection to a mock version of Snowflake."""
        self._mock_cursor = MockCursor()

    def cursor(self):
        return self._mock_cursor


@pytest.fixture()
def executed_queries():
    return []


@pytest.fixture(autouse=True)
def cache_dir(monkeypatch, tmpdir) -> Path:
    """Return path to monkeypatched query cache directory."""
    tmpdir = Path(tmpdir)
    monkeypatch.setattr(sources, "DEFAULT_CACHE_DIRECTORY", tmpdir)
    return tmpdir


@pytest.fixture(autouse=True)
def source(cache_dir, monkeypatch, request, executed_queries) -> Optional[MockCursor]:
    """Return mocked snowflake cursor."""

    def sql_to_arrow(query: str) -> pa.Table:
        executed_queries.append(query)
        return pa.Table.from_pydict({"column": [1, 2, 3]})

    source = sources.DataSource.from_query_function(
        sql_to_arrow,
        cache_directory=cache_dir,
        default_cache_ttl=timedelta(days=1000),
    )
    return source


def test_sql_to_arrow(source, monkeypatch):
    """It should return an Arrow Table."""

    # We freeze the time during the execution of this test
    class FrozenDatetime:
        @classmethod
        def now(cls):
            return datetime(year=1903, month=12, day=28)

    monkeypatch.setattr(sources, "datetime", FrozenDatetime)

    @source.query()
    def dummy():
        return "some query"

    # Execute a query that always will be valid
    arrow_table = dummy.as_arrow()

    assert arrow_table.to_pydict() == {"column": [1, 2, 3]}

    # Some additional metadata is also inserted into the arrow table
    assert arrow_table.schema.metadata == {
        b"sql_query": b"some query",
        b"query_start_time": b"1903-12-28T00:00:00",
        b"query_finish_time": b"1903-12-28T00:00:00",
    }


def test_uncached_query(source, executed_queries, cache_dir: Path):
    """It should not cache queries by default."""

    @source.query()
    def products():
        return "query"

    # First time it is called we should execute the query
    products()
    assert executed_queries == ["query"]
    # And no cache file is created
    assert not any(cache_dir.iterdir())

    # The next time the query is executed again
    products()
    assert executed_queries == ["query", "query"]
    # And still no cache file
    assert not any(cache_dir.iterdir())


def test_cached_query(source, executed_queries: MockCursor, cache_dir: Path):
    """It should cache queries if so parametrized."""

    # We enable cache for the given query
    @source.query(cache=True)
    def products(version: int):
        return f"query {version}"

    # The cache is stored in the "products" sub-folder
    cache_dir = cache_dir / "products"

    # First time the query is executed
    products(version=1)
    assert executed_queries == ["query 1"]
    # And the result is stored in a cache file
    assert len(list(cache_dir.iterdir())) == 1

    # The next time the query is *not* executed
    products(version=1)
    assert executed_queries == ["query 1"]
    # And the cache file persists
    assert len(list(cache_dir.iterdir())) == 1

    # But if we change the query itself, it is executed
    products(version=2)
    assert executed_queries == ["query 1", "query 2"]
    # And it is cached in a separate file
    assert len(list(cache_dir.iterdir())) == 2

    # If we delete the cache file, the query is re-executed
    for cache_file in cache_dir.iterdir():
        cache_file.unlink()
    products(version=1)
    assert executed_queries == ["query 1", "query 2", "query 1"]
    # And the cache file is rewritten
    assert len(list(cache_dir.iterdir())) == 1

    # We clear the cache with .clear_cache()
    products.refresh_cache(version=1)
    assert executed_queries == ["query 1", "query 2", "query 1", "query 1"]
    # We can also clear caches that have never existed
    products.refresh_cache(version=3)
    assert executed_queries[-1] == "query 3"


def test_cached_query_with_explicit_path(
    source, executed_queries, cache_dir: Path, tmpdir
) -> None:
    """It should cache queries in the provided path."""
    cache_path = Path(tmpdir / "name.parquet")

    # This time we specify an explicit path
    @source.query(cache=cache_path)
    def products(version):
        return f"query {version}"

    # At first the path does not exist
    assert not cache_path.exists()

    # We then execute and cache the query
    products(version=1)
    assert cache_path.exists()
    assert executed_queries == ["query 1"]

    # And the next time it is reused
    products(version=1)
    assert executed_queries == ["query 1"]
    assert cache_path.exists()

    # If the query changes, it is re-executed
    products(version=2)
    assert executed_queries == ["query 1", "query 2"]


def test_cached_query_with_relative_path(source, cache_dir: Path) -> None:
    """Relative paths should be interpreted relative to the cache directory."""
    relative_path = Path("foo/bar.parquet")

    @source.query(cache=relative_path)
    def products():
        return "query"

    products()
    assert (cache_dir / "foo" / "bar.parquet").exists()


def test_cached_query_with_format_string(source, cache_dir: Path) -> None:
    """Strings with placeholders should be interpolated."""

    @source.query(cache="version-{version}.parquet")
    def products(version: int):
        return f"query {version}"

    # It should work for both positional arguments...
    products(1)
    assert (cache_dir / "version-1.parquet").exists()
    # ... and keywords
    products(version=2)
    assert (cache_dir / "version-2.parquet").exists()


def test_cache_ttl(source, monkeypatch, executed_queries):
    """It should automatically refresh the cache according to the TTL."""

    # We freeze the time during the execution of this test
    class FrozenDatetime:
        def __init__(self, year: int, month: int, day: int) -> None:
            self.frozen_time = datetime(year=year, month=month, day=day)
            monkeypatch.setattr(sources, "datetime", self)

        def now(self):
            return self.frozen_time

        @staticmethod
        def fromisoformat(*args, **kwargs):
            return datetime.fromisoformat(*args, **kwargs)

    # The cache should be cleared every week
    @source.query(cache=True, ttl=timedelta(weeks=1))
    def users():
        return "query"

    # The first time the query should be executed
    FrozenDatetime(year=2000, month=1, day=1)
    users()
    assert executed_queries == ["query"]

    # The next time it should not be executed
    users()
    assert executed_queries == ["query"]

    # Even if we advance the time by one day,
    # the cache should still be used.
    FrozenDatetime(year=2000, month=1, day=2)
    users()
    assert executed_queries == ["query"]

    # Then we let one week pass, and the cache should be cleared
    FrozenDatetime(year=2000, month=1, day=8)
    users()
    assert executed_queries == ["query", "query"]

    # But then it will be reused for another week
    users()
    assert executed_queries == ["query", "query"]
