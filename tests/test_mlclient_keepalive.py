"""HttpClient survives the server closing an idle keep-alive connection (ml-service «flapping»)."""

from __future__ import annotations

import asyncio

import pytest

from backend.mlclient import HttpClient, HttpError

RESPONSE = b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\n{"ok":true}'


async def _serve_once_then_close(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Answer one request, then close the connection like uvicorn after its keep-alive timeout."""
    while (await reader.readline()) not in (b"\r\n", b""):
        pass
    writer.write(RESPONSE)
    await writer.drain()
    writer.close()


def test_retries_once_when_server_closed_idle_connection() -> None:
    async def scenario() -> list[int]:
        server = await asyncio.start_server(_serve_once_then_close, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HttpClient(f"http://127.0.0.1:{port}")
        statuses = []
        try:
            for _ in range(3):
                status, body = await client.request("GET", "/health")
                assert body == b'{"ok":true}'
                statuses.append(status)
                await asyncio.sleep(0.05)  # the server has closed the connection by now
        finally:
            client.close()
            server.close()
            await server.wait_closed()
        return statuses

    assert asyncio.run(scenario()) == [200, 200, 200]


async def _read_then_close(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Take the request, answer nothing. The request is read first: closing a socket with unread data sends
    a TCP reset, and the client would see ``ConnectionResetError`` or the EOF depending on timing."""
    while (await reader.readline()) not in (b"\r\n", b""):
        pass
    writer.close()


def test_fresh_connection_failure_is_not_retried() -> None:
    async def scenario() -> None:
        server = await asyncio.start_server(_read_then_close, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HttpClient(f"http://127.0.0.1:{port}")
        try:
            await client.request("GET", "/health")
        finally:
            client.close()
            server.close()
            await server.wait_closed()

    with pytest.raises(HttpError, match="connection closed before the response"):
        asyncio.run(scenario())
