# -*- coding: utf-8 -*-
import unittest

import pytest


@pytest.mark.tox
class Testmmvae(unittest.TestCase):
    """Basic tests for the MMVAE_mst package."""

    def test_import(self):
        """Tests the imports for the MMVAE_mst package."""
        import mmvae_hub
        import mmvae_hub.base
