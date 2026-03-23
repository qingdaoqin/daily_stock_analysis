# -*- coding: utf-8 -*-
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import feishu_doc


class TestFeishuDocOptionalSdk(unittest.TestCase):
    @patch("src.feishu_doc.get_config")
    def test_missing_sdk_skips_doc_creation_gracefully(self, mock_get_config):
        mock_get_config.return_value = SimpleNamespace(
            feishu_app_id="app-id",
            feishu_app_secret="app-secret",
            feishu_folder_token="folder-token",
        )

        with patch.object(feishu_doc, "_LARK_SDK_AVAILABLE", False):
            manager = feishu_doc.FeishuDocManager()

        self.assertFalse(manager.is_configured())
        self.assertIsNone(manager.create_daily_doc("test", "# hello"))


if __name__ == "__main__":
    unittest.main()
