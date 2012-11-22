"""Partial-PP unit tests."""

from partial_pp import *
import unittest

class TestCommandLineParsing(unittest.TestCase):
    def test_defn(self):
        args = ['-DFOO']
        defines = parse_args(args)
        self.assertEqual(defines, (['FOO'],[]))

    def test_eq(self):
        args = ['-DFOO=BAR']
        defines = parse_args(args)
        self.assertEqual(defines, ([('FOO', 'BAR')],[]))

    def test_undef(self):
        args = ['-UFOO']
        defines = parse_args(args)
        self.assertEqual(defines, ([], ['FOO']))

    def test_errors(self):
        with self.assertRaises(InvalidCommandLineArgError):
            parse_args(['FOO'])
        with self.assertRaises(InvalidCommandLineArgError):
            parse_args(['-UFOO=BAR'])

class TestSimplification(unittest.TestCase):
    def setUp(self):
        defines = ['DEFINED', ('ONE', 1), ('ZERO', 0)]
        undefines = ['UNDEFINED']
        self.simplifier = Simplifier(defines, undefines)

    def test_constants(self):
        self.assertEqual(self.simplifier.simplify('1'), True)
        self.assertEqual(self.simplifier.simplify('0'), False)
        self.assertEqual(self.simplifier.simplify('5'), True)
        self.assertEqual(self.simplifier.simplify('true'), True)
        self.assertEqual(self.simplifier.simplify('false'), False)

    def test_defined(self):
        self.assertEqual(self.simplifier.simplify('defined(DEFINED)'), True)
        self.assertEqual(self.simplifier.simplify('defined DEFINED'), True)

        self.assertEqual(self.simplifier.simplify('defined(UNDEFINED)'), False)
        self.assertEqual(self.simplifier.simplify('defined UNDEFINED'), False)

        self.assertEqual(self.simplifier.simplify('defined(FOO)'),
                                                  'defined(FOO)')
        self.assertEqual(self.simplifier.simplify('defined FOO'),
                                                  'defined FOO')

class TestTokenizer(unittest.TestCase):
    def test_defined(self):
        expr = 'defined(FOO)'
        toks = [(Token.DEFINED, 'defined'),
                (Token.LPAREN, '(',),
                (Token.IDENTIFIER, 'FOO'),
                (Token.RPAREN, ')'),
               ]
        self.assertEqual([tok.to_tuple() for tok in Tokenizer(expr)], toks)

        expr = 'defined( FOO)'
        toks = [(Token.DEFINED, 'defined'),
                (Token.LPAREN, '(',),
                (Token.IDENTIFIER, ' FOO'),
                (Token.RPAREN, ')'),
               ]
        self.assertEqual([tok.to_tuple() for tok in Tokenizer(expr)], toks)

    def test_empty(self):
        self.assertEqual([tok for tok in Tokenizer('')], [])

    def test_true_false(self):
        true_tok = Tokenizer('true').next()
        self.assertEqual(true_tok.kind, Token.TRUE)

        false_tok = Tokenizer('false').next()
        self.assertEqual(false_tok.kind, Token.FALSE)

    def test_operators(self):
        expr = '&& & || | ! != =='
        toks = [(Token.LOGICAL_AND, '&&'),
                (Token.BITWISE_AND, ' &'),
                (Token.LOGICAL_OR, ' ||'),
                (Token.BITWISE_OR, ' |'),
                (Token.LOGICAL_NOT, ' !'),
                (Token.NEQ, ' !='),
                (Token.EQ, ' =='),
               ]
        self.assertEqual([tok.to_tuple() for tok in Tokenizer(expr)], toks)

if __name__ == '__main__':
    unittest.main()
