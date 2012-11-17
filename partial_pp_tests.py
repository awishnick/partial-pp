"""Partial-PP unit tests."""

import partial_pp
import unittest

class TestCommandLineParsing(unittest.TestCase):
    def test_defn(self):
        args = ['-DFOO']
        defines = partial_pp.extract_defines(args)
        self.assertEqual(defines, ['FOO'])

    def test_eq(self):
        args = ['-DFOO=BAR']
        defines = partial_pp.extract_defines(args)
        self.assertEqual(defines, [('FOO', 'BAR')])

if __name__ == '__main__':
    unittest.main()
