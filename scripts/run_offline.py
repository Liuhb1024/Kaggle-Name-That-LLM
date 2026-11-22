"""Run the CLI with Python network connections and DNS disabled in this process.

This is a reproducibility check, not an OS sandbox for untrusted external code.
"""

import socket
from typing import NoReturn


def deny_network(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("network access is disabled for this baseline process")


if __name__ == "__main__":
    socket.socket.connect = deny_network
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network

    from ntl_baseline.cli import main

    raise SystemExit(main())
