"""Serial smoke test; opt in with SERIAL_PORT=/dev/ttyUSB0."""

import os

import pytest


@pytest.mark.skipif(not os.getenv("SERIAL_PORT"), reason="SERIAL_PORT is not configured")
def test_serial_port_opens() -> None:
    import serial

    connection = serial.Serial(os.environ["SERIAL_PORT"], baudrate=115200, timeout=1)
    connection.close()
