import hashlib
import inspect
import tempfile
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, Optional, Protocol, Type, Union

import polars as pl
import pyarrow as pa
from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from patito import Model

# Caches are set to last infinitely long if not specified
DEFAULT_TTL = timedelta(weeks=52 * 100)

# Put cache files in the OS-wide temporary directory
DEFAULT_CACHE_DIRECTORY = Path(tempfile.gettempdir())

P = ParamSpec("P")


class QueryConstructor(Protocol[P]):
    """A function taking arbitrary arguments and returning an SQL query string."""

    __name__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """Return SQL query constructed from the given parameters."""
        ...


class QueryExecutor(Generic[P]):
    """A class acting as a function that returns a polars.DataFrame when called."""

    def __init__(
        self,
        query_constructor: QueryConstructor[P],
        query_executor: Callable[[str], pa.Table],
        cache_directory: Path = DEFAULT_CACHE_DIRECTORY,
        cache: Union[str, Path, bool] = False,
        ttl: timedelta = DEFAULT_TTL,
        model: Union[Type["Model"], None] = None,
    ) -> None:
        """Convert SQL string query function to polars.DataFrame function.

        Args:
            query_constructor: A function that takes arbitrary arguments and returns
                an SQL query string.
            query_constructor: A function that takes in a query string and returns
                an arrow table.
            cache_directory: Path to directory to be used for query caches.
            cache: See DataSource.query for documentation.
            ttl: See DataSource.query for documentation.
            model: See DataSource.query for documentation.
        """
        if not isinstance(cache, bool) and Path(cache).suffix != ".parquet":
            raise ValueError("Cache paths must have the '.parquet' file extension!")

        self._cache = cache
        self._wrapped_function = query_constructor
        self._query_executor = query_executor
        self._cache_directory = cache_directory

        # We construct the new function with the same parameter signature as
        # query_constructor, but with polars.DataFrame as the return type.
        @wraps(query_constructor)
        def cached_func(*args: P.args, **kwargs: P.kwargs) -> pl.DataFrame:
            sql_query = query_constructor(*args, **kwargs)
            cache_path = self.cache_path(*args, **kwargs)
            if cache_path and cache_path.exists():
                metadata = pa.parquet.read_schema(cache_path).metadata

                # Check if the cache file was produced by an identical SQL query
                is_same_query = metadata.get(b"sql_query") == sql_query.encode("utf-8")

                # Check if the cache is too old to be re-used
                cache_created_time = datetime.fromisoformat(
                    metadata.get(
                        b"query_start_time", b"1900-0-0T00:00:00.000000"
                    ).decode("utf-8")
                )
                is_fresh_cache = (datetime.now() - cache_created_time) < ttl

                if is_same_query and is_fresh_cache:
                    # We do not use polars.read_parquet because it does not handle
                    # Decimal-encoded columns directly, while pyarrow.parquet.read_table
                    # + polars.from_arrow handles it fine.
                    arrow_table: pa.lib.Table = pa.parquet.read_table(  # noqa: PDO12
                        cache_path
                    )
                    return pl.from_arrow(arrow_table)

            arrow_table = self.as_arrow(*args, **kwargs)
            if cache_path:
                pa.parquet.write_table(table=arrow_table, where=cache_path)

            polars_df = pl.from_arrow(arrow_table)
            if model:
                model.validate(polars_df)
            return polars_df

        self._cached_func = cached_func

    def as_arrow(self, *args: P.args, **kwargs: P.kwargs) -> pa.lib.Table:
        """Execute SQL query in and return the result as an arrow table.

        Args:
            args: Positional arguments to construct query.
            kwargs: Keyword arguments to construct query.

        Returns:
            A pyarrow.Table object representing the result of the query.
        """
        sql_query = self.sql_query(*args, **kwargs)

        start_time = datetime.now()
        arrow_table = self._query_executor(sql_query)
        finish_time = datetime.now()

        # Store additional metadata which is useful when the arrow table is written to a
        # parquet file as a caching mechanism.
        metadata: dict = arrow_table.schema.metadata or {}
        metadata.update(
            {
                "sql_query": sql_query,
                "query_start_time": start_time.isoformat(),
                "query_finish_time": finish_time.isoformat(),
            }
        )
        return arrow_table.replace_schema_metadata(metadata)

    def cache_path(self, *args: P.args, **kwargs: P.kwargs) -> Optional[Path]:
        """Return the deterministic cache path for the given parameters.

        Args:
            args: The positional arguments passed to the query constructor.
            kwargs: The keyword arguments passed to the query constructor.

        Returns: A deterministic path to a parquet cache. None if caching is disabled.
        """
        if isinstance(self._cache, Path):
            # Interpret relative paths relative to the main query cache directory
            cache_path = self._cache_directory.joinpath(self._cache)
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            return cache_path
        elif isinstance(self._cache, str):
            # We convert args+kwargs to kwargs-only and use it to format the string
            function_signature = inspect.signature(self._wrapped_function)
            bound_arguments = function_signature.bind(*args, **kwargs)
            return self._cache_directory / self._cache.format(
                **bound_arguments.arguments
            )
        elif self._cache is True:
            directory: Path = self._cache_directory / self._wrapped_function.__name__
            directory.mkdir(exist_ok=True, parents=True)
            sql_query = self.sql_query(*args, **kwargs)
            sql_query_hash = hashlib.sha1(
                sql_query.encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()
            return directory / f"{sql_query_hash}.parquet"
        else:
            return None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> pl.DataFrame:
        return self._cached_func(*args, **kwargs)

    def sql_query(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """Return SQL query to be executed for the given parameters."""
        return self._wrapped_function(*args, **kwargs)

    def refresh_cache(self, *args: P.args, **kwargs: P.kwargs) -> pl.DataFrame:
        """Force query execution by refreshing the cache."""
        cache_path = self.cache_path(*args, **kwargs)
        if cache_path and cache_path.exists():
            cache_path.unlink()
        return self._cached_func(*args, **kwargs)


class DataSource:
    def __init__(
        self,
        query_executor: Callable[[str], pa.Table],
        cache_directory: Path,
        default_cache_ttl: timedelta,
    ) -> None:
        self.query_executor = query_executor
        self.cache_directory = cache_directory
        self.default_cache_ttl = default_cache_ttl

    @classmethod
    def from_query_executor(
        cls,
        query_executor: Callable[[str], pa.Table],
        cache_directory: Path,
        default_cache_ttl: timedelta,
    ) -> "DataSource":
        """
        Construct data source from SQL query executor.

        Args:
            query_executor: A function taking an SQL query string and returning
                an arrow table.
            cache_directory: A path to a directory to store parquet files for
                caching queries.
        """
        return cls(
            query_executor=query_executor,
            cache_directory=cache_directory,
            default_cache_ttl=default_cache_ttl,
        )

    def query(
        self,
        cache: Union[str, Path, bool] = False,
        ttl: Optional[timedelta] = None,
        model: Union[Type["Model"], None] = None,
    ) -> Callable[[QueryConstructor[P]], QueryExecutor[P]]:
        """Execute the returned query string and return a polars dataframe.

        Args:
            cache: If queries should be cached in order to save time and costs.
                The cache will only be used if the exact same SQL string has
                been executed before.
                If the parameter is specified as ``True``, a parquet file is
                created for each unique query string, and is located at:
                    artifacts/query_cache/<function_name>/<query_md5_hash>.parquet
                If the a string or ``pathlib.Path`` object is provided, the given path
                will be used, but it must have a '.parquet' file extension.
                Relative paths are interpreted relative to artifacts/query_cache/
                in the workspace root. The given parquet path will be overwritten
                if the query string changes, so only the latest query string value
                will be cached.
            ttl: The Time to Live (TTL) of the cache specified as a datetime.timedelta
                object. When the cache becomes older than the specified TTL, the query
                will be re-executed on the next invocation of the query function
                and the cache will refreshed.
            model: An optional Patito model used to validate the content of the
                dataframe before return.
        Return: A new function which returns a polars DataFrame based on the query
            specified by the original function's return string.
        """

        def wrapper(query_constructor: QueryConstructor) -> QueryExecutor:
            return QueryExecutor(
                query_constructor=query_constructor,
                query_executor=self.query_executor,
                cache=cache,
                ttl=ttl if ttl is not None else self.default_cache_ttl,
                model=model,
                cache_directory=self.cache_directory,
            )

        return wrapper
