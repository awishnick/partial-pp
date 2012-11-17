"""Partial-PP unit tests."""

import partial_pp
import unittest

class TestCommandLineParsing(unittest.TestCase):
    def test_defn(self):
        args = ['-DFOO']
        defines = partial_pp.parse_args(args)
        self.assertEqual(defines, (['FOO'],[]))

    def test_eq(self):
        args = ['-DFOO=BAR']
        defines = partial_pp.parse_args(args)
        self.assertEqual(defines, ([('FOO', 'BAR')],[]))

    def test_undef(self):
        args = ['-UFOO']
        defines = partial_pp.parse_args(args)
        self.assertEqual(defines, ([], ['FOO']))

    def test_errors(self):
        with self.assertRaises(partial_pp.InvalidCommandLineArgError):
            partial_pp.parse_args(['FOO'])
        with self.assertRaises(partial_pp.InvalidCommandLineArgError):
            partial_pp.parse_args(['-UFOO=BAR'])

if __name__ == '__main__':
    unittest.main()
