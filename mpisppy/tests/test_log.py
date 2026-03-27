###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for mpisppy/log.py logging utilities."""

import logging
import os
import sys
import tempfile
import unittest

from mpisppy.log import setup_logger

# Names used by tests -- cleaned up in tearDown
_TEST_LOGGER_NAMES = [
    "mpisppy_test_log_stdout",
    "mpisppy_test_log_stderr",
    "mpisppy_test_log_file",
    "mpisppy_test_log_fmt",
    "mpisppy_test_log_noprop",
    "mpisppy_test_log_ret",
    "mpisppy_test_log_wmode",
]


def _cleanup_logger(name):
    """Remove all handlers from a named logger."""
    log = logging.getLogger(name)
    for h in list(log.handlers):
        try:
            h.close()
        except Exception:
            pass
        log.removeHandler(h)


class TestSetupLogger(unittest.TestCase):
    """Tests for the setup_logger() function."""

    def tearDown(self):
        for name in _TEST_LOGGER_NAMES:
            _cleanup_logger(name)

    # ------------------------------------------------------------------
    # setup_logger() does NOT return the logger -- callers use
    # logging.getLogger(name) after the call.
    # ------------------------------------------------------------------

    def test_stream_handler_stdout(self):
        name = "mpisppy_test_log_stdout"
        setup_logger(name, sys.stdout, level=logging.DEBUG)
        log = logging.getLogger(name)
        self.assertIsInstance(log, logging.Logger)
        self.assertEqual(log.level, logging.DEBUG)

    def test_stream_handler_stderr(self):
        name = "mpisppy_test_log_stderr"
        setup_logger(name, sys.stderr, level=logging.WARNING)
        log = logging.getLogger(name)
        self.assertEqual(log.level, logging.WARNING)
        handlers = [h for h in log.handlers
                    if isinstance(h, logging.StreamHandler)
                    and h.stream is sys.stderr]
        self.assertGreaterEqual(len(handlers), 1)

    def test_file_handler(self):
        name = "mpisppy_test_log_file"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            fname = f.name
        try:
            setup_logger(name, fname, level=logging.INFO)
            log = logging.getLogger(name)
            log.info("hello log")
            for h in list(log.handlers):
                h.flush()
                h.close()
                log.removeHandler(h)
            with open(fname) as fh:
                content = fh.read()
            self.assertIn("hello log", content)
        finally:
            os.unlink(fname)

    def test_custom_format(self):
        name = "mpisppy_test_log_fmt"
        fmt = "%(levelname)s - %(message)s"
        setup_logger(name, sys.stdout, level=logging.DEBUG, fmt=fmt)
        log = logging.getLogger(name)
        formatters = [h.formatter._fmt for h in log.handlers
                      if h.formatter is not None]
        self.assertIn(fmt, formatters)

    def test_no_propagate(self):
        name = "mpisppy_test_log_noprop"
        setup_logger(name, sys.stdout)
        log = logging.getLogger(name)
        self.assertFalse(log.propagate)

    def test_handler_added(self):
        name = "mpisppy_test_log_ret"
        setup_logger(name, sys.stdout, level=logging.DEBUG)
        log = logging.getLogger(name)
        self.assertGreaterEqual(len(log.handlers), 1)

    def test_write_mode(self):
        name = "mpisppy_test_log_wmode"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            fname = f.name
        try:
            setup_logger(name, fname, mode='w')
            log = logging.getLogger(name)
            log.debug("first write")
            _cleanup_logger(name)

            # Re-open in write mode to overwrite
            setup_logger(name, fname, mode='w')
            log = logging.getLogger(name)
            log.debug("second write")
            _cleanup_logger(name)

            with open(fname) as fh:
                content = fh.read()
            # mode='w' overwrites, so only the second write should be present
            self.assertIn("second write", content)
            self.assertNotIn("first write", content)
        finally:
            os.unlink(fname)


class TestRootLogger(unittest.TestCase):
    """Tests that the mpisppy root logger is configured at module level."""

    def test_root_logger_exists(self):
        root_log = logging.getLogger("mpisppy")
        self.assertIsNotNone(root_log)

    def test_root_logger_level_info(self):
        root_log = logging.getLogger("mpisppy")
        self.assertEqual(root_log.level, logging.INFO)

    def test_root_logger_has_handler(self):
        root_log = logging.getLogger("mpisppy")
        self.assertGreater(len(root_log.handlers), 0)

    def test_root_logger_handler_is_stream(self):
        root_log = logging.getLogger("mpisppy")
        stream_handlers = [h for h in root_log.handlers
                           if isinstance(h, logging.StreamHandler)]
        self.assertGreater(len(stream_handlers), 0)


if __name__ == "__main__":
    unittest.main()
