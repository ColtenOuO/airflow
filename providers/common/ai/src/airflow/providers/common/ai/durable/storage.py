# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""ObjectStorage-backed durable storage for pydantic-ai agent step caching."""

from __future__ import annotations

import contextlib
import hashlib
import json
from functools import lru_cache
from typing import Any

import structlog
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse

# Sentinel to distinguish "cached None" from "no cache entry" for tool results.
# Shared with the task state store backend so the envelope shape cannot drift.
from airflow.providers.common.ai.durable.base import TOOL_RESULT_SENTINEL as _SENTINEL

log = structlog.get_logger(logger_name="task")

SECTION = "common.ai"


@lru_cache(maxsize=1)
def _get_base_path():
    from airflow.providers.common.compat.sdk import conf
    from airflow.sdk import ObjectStoragePath

    path = conf.get(SECTION, "durable_cache_path", fallback="")
    if not path:
        raise ValueError(
            "durable=True requires [common.ai] durable_cache_path to be set. "
            "Example: durable_cache_path = file:///tmp/airflow_durable_cache"
        )
    return ObjectStoragePath(path)


class DurableStorage:
    """
    Stores step-level caches as one file per key on ObjectStorage.

    Each model response and tool result is written to its own file under
    ``{base_path}/{cache_id}/{key}.json``, where ``cache_id`` is a hash of
    the task instance's identity (dag, task, run, map index) so distinct
    task instances never share a cache directory. Writing each step to its
    own file -- rather than rewriting one growing JSON blob on every save --
    keeps each save proportional to that one entry instead of the whole
    history so far.

    The directory survives Airflow task retries since it lives outside the
    XCom system.  It is deleted on successful task completion.

    :param dag_id: DAG ID of the running task.
    :param task_id: Task ID of the running task.
    :param run_id: DAG run ID.
    :param map_index: Map index for mapped tasks (``-1`` for non-mapped).
    """

    def __init__(
        self,
        *,
        dag_id: str,
        task_id: str,
        run_id: str,
        map_index: int = -1,
    ) -> None:
        # Hash the identity components with a separator that cannot appear in
        # them, so distinct task instances can never alias to the same cache
        # directory. A plain ``_``-joined string collides -- e.g. dag ``etl`` +
        # task ``load_data`` and dag ``etl_load`` + task ``data`` both yield
        # ``etl_load_data`` -- letting one task read, overwrite, or delete
        # another's durable cache.
        identity = "\x00".join([dag_id, task_id, run_id, str(map_index)])
        self._cache_id = hashlib.sha256(identity.encode()).hexdigest()

    def _get_dir(self):
        return _get_base_path() / self._cache_id

    def _get_entry_path(self, key: str):
        return self._get_dir() / f"{key}.json"

    def _read_entry(self, key: str) -> Any | None:
        """Load and JSON-decode a single entry's file. Returns ``None`` on a miss."""
        try:
            return json.loads(self._get_entry_path(key).read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
            return None

    def _write_entry(self, key: str, value: Any) -> None:
        """JSON-encode and persist a single entry to its own file."""
        path = self._get_entry_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def save_model_response(self, key: str, response: ModelResponse, *, fingerprint: str | None) -> None:
        """Serialize and store a ModelResponse with the request fingerprint that produced it."""
        # Store the dumped messages as native JSON-compatible objects, not a
        # pre-encoded string: the entry is JSON-encoded once in ``_write_entry``,
        # so embedding a string here would double-encode the (large) response payload.
        self._write_entry(
            key,
            {
                "fingerprint": fingerprint,
                "data": ModelMessagesTypeAdapter.dump_python([response], mode="json"),
            },
        )

    def load_model_response(self, key: str) -> tuple[ModelResponse | None, str | None]:
        """
        Load a cached ModelResponse and its stored request fingerprint.

        Returns ``(None, None)`` if not cached. Entries written before
        fingerprints existed load with a ``None`` fingerprint.
        """
        raw = self._read_entry(key)
        if raw is None:
            return None, None
        try:
            if isinstance(raw, dict):
                messages = ModelMessagesTypeAdapter.validate_python(raw["data"])
                fingerprint = raw.get("fingerprint")
            else:
                # Legacy entry: the adapter JSON (a list) was stored directly as a string.
                messages = ModelMessagesTypeAdapter.validate_json(raw)
                fingerprint = None
        except (KeyError, IndexError, ValueError):
            # A torn or malformed entry degrades to a miss (the step re-runs),
            # never a task crash -- the cache is best-effort.
            log.warning("Durable: ignoring malformed cached model response", key=key)
            return None, None
        if not messages:
            return None, None
        return messages[0], fingerprint  # type: ignore[return-value]

    def save_tool_result(self, key: str, result: Any, *, fingerprint: str | None) -> None:
        """
        Store a tool call result with the call fingerprint that produced it.

        Non-serializable results (e.g. BinaryContent from MCP tools) are
        skipped with a warning -- the tool call still succeeds, but won't
        be replayed on retry.
        """
        try:
            # Probe serializability before writing: a non-serializable result
            # must skip only this entry, not raise out of the tool call.
            # TypeError covers unsupported types; ValueError covers circular references.
            json.dumps(result)
        except (TypeError, ValueError):
            log.warning(
                "Durable: skipping cache for non-serializable tool result",
                key=key,
                type=type(result).__name__,
            )
            return
        self._write_entry(key, {_SENTINEL: True, "value": result, "fingerprint": fingerprint})

    def load_tool_result(self, key: str) -> tuple[bool, Any, str | None]:
        """
        Load a cached tool result and its stored call fingerprint.

        Returns a (found, value, fingerprint) tuple since the cached value
        itself could be None. Entries written before fingerprints existed
        load with a ``None`` fingerprint.
        """
        raw = self._read_entry(key)
        if raw is None:
            return False, None, None
        # Legacy entries were stored as a JSON string; new entries are native dicts.
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict) or _SENTINEL not in raw:
            return False, None, None
        return True, raw["value"], raw.get("fingerprint")

    def cleanup(self) -> None:
        """Delete the cache directory after successful execution."""
        # Best-effort cleanup
        with contextlib.suppress(FileNotFoundError, OSError):
            directory = self._get_dir()
            for entry in directory.iterdir():
                with contextlib.suppress(FileNotFoundError, OSError):
                    entry.unlink()
            directory.rmdir()
        log.debug("Durable cache cleaned up")
