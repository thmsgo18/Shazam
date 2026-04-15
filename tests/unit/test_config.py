from __future__ import annotations

import unittest
from unittest.mock import patch

import src.config as config


class ConfigTests(unittest.TestCase):
    def test_get_collection_key_uses_default_method(self) -> None:
        with patch.object(config, "EMBEDDING_METHOD", "mfcc"):
            self.assertEqual(config.get_collection_key(), "mfcc")

    def test_get_collection_key_sanitizes_clap_model_name(self) -> None:
        with patch.object(config, "CLAP_MODEL_NAME", "laion/larger-clap-music"):
            self.assertEqual(config.get_collection_key("clap"), "clap_larger_clap_music")

    def test_get_collection_key_supports_muq_and_mert(self) -> None:
        with patch.object(config, "MUQ_MODEL_NAME", "OpenMuQ/MuQ-large-msd-iter"), \
             patch.object(config, "MERT_MODEL_NAME", "m-a-p/MERT-v1-95M"):
            self.assertEqual(config.get_collection_key("muq"), "muq_MuQ_large_msd_iter")
            self.assertEqual(config.get_collection_key("mert"), "mert_MERT_v1_95M")


if __name__ == "__main__":
    unittest.main()
