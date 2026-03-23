# -*- coding: utf-8 -*-
"""Shared lightweight test stubs for optional dependencies."""

import sys
from unittest.mock import MagicMock


if "fake_useragent" not in sys.modules:
    ua_instance = MagicMock()
    ua_instance.random = "Mozilla/5.0"
    ua_instance.chrome = "Mozilla/5.0"

    fake_useragent_module = MagicMock()
    fake_useragent_module.UserAgent = MagicMock(return_value=ua_instance)
    sys.modules["fake_useragent"] = fake_useragent_module
