"""Partially preprocess a C/C++ source file, only acting on certain preprocessor
definitions.
"""

import re
import string

def build_defines_map(defines):
    """Take a list of defines, and build it into a map."""
    defines_map = {}

    for define in defines:
        if isinstance(define, tuple):
            (symbol, definition) = define
            defines_map[symbol] = definition
        else:
            assert isinstance(define, str)
            defines_map[define] = ''

    return defines_map

class Token:
    """A single preprocessor token."""

    UNKNOWN = 0
    IDENTIFIER = 1
    NUMERIC_LITERAL = 2
    LPAREN = 3
    RPAREN = 4
    DEFINED = 5
    TRUE = 7
    FALSE = 8
    BITWISE_AND = 9
    LOGICAL_AND = 10
    BITWISE_OR = 11
    LOGICAL_OR = 12
    LOGICAL_NOT = 13
    NEQ = 14
    EQ = 15
    
    BINARY_OPERATORS = {
        BITWISE_AND,
        LOGICAL_AND,
        BITWISE_OR,
        LOGICAL_OR,
        NEQ,
        EQ,
    }

    UNARY_OPERATORS = {
        LOGICAL_NOT
    }

    OPERATOR_PRECEDENCES = {
        LOGICAL_NOT: 3,
        EQ: 9,
        NEQ: 9,
        BITWISE_AND: 10,
        BITWISE_OR: 12,
        LOGICAL_AND: 13,
        LOGICAL_OR: 14,
    }

    def __init__(self, kind, val, start, end):
        self.kind = kind
        self.val = val
        self.start = start
        self.end = end

    def __repr__(self):
        return '({}, {})'.format(self.kind, self.val)

    def to_tuple(self):
        """Convert to a tuple of (kind, val)"""
        return (self.kind, self.val)

    def get_leading_whitespace(self):
        """Return the leading whitespace, or an empty string if none."""
        m = re.match('^\s+', self.val)
        if m is None:
            return ''
        return m.group(0)

    def is_binary_operator(self):
        """Return if this is a binary operator."""

        return self.kind in Token.BINARY_OPERATORS

    def is_unary_operator(self):
        """Return if this is a unary operator."""

        return self.kind in Token.UNARY_OPERATORS

    def get_operator_precedence(self):
        """Return this operator's precedence.

        If this is not an operator, return None.
        """

        if self.kind in Token.OPERATOR_PRECEDENCES:
            return Token.OPERATOR_PRECEDENCES[self.kind]

        return None


class TokenizerError(Exception):
    def __init__(self, msg, pos):
        self.msg = msg
        self.pos = pos

    def __repr__(self):
        return 'Syntax error (pos = {}): {}'.format(pos, msg)

class Tokenizer:
    """Tokenize preprocessor expressions."""

    def __init__(self, expr):
        self.expr = expr
        self.length = len(expr)
        self.pos = 0

    def __iter__(self):
        return self

    def consume_whitespace(self):
        """Advance pos until it's not pointing at whitespace."""
        if self.pos >= self.length:
            return

        if self.expr[self.pos] not in string.whitespace:
            return

        begin = self.pos
        while self.pos < self.length:
            if self.expr[self.pos] not in string.whitespace:
                return
            self.pos += 1

    # These are single-character tokens that are not the prefix of any other
    # token.
    SINGLE_CHAR_TOKS = {'(': Token.LPAREN,
                        ')': Token.RPAREN,
                       }

    # These are operators that can either be one or two characters.
    SINGLE_OR_DOUBLE_CHAR_TOKS = {
        '&': (Token.BITWISE_AND, Token.LOGICAL_AND),
        '|': (Token.BITWISE_OR, Token.LOGICAL_OR),
    }

    def try_lex_operator(self):
        """Tokenize operators or parentheses if they're present.

        Return the token if present and update pos, otherwise update nothing and
        return None.
        """

        if self.expr[self.pos] in self.SINGLE_CHAR_TOKS:
            tok = self.expr[self.pos]
            self.pos += 1
            return (self.SINGLE_CHAR_TOKS[tok], tok)

        if self.expr[self.pos] in self.SINGLE_OR_DOUBLE_CHAR_TOKS:
            tok = self.expr[self.pos]
            toklen = 1
            self.pos += 1

            if self.pos < self.length:
                if self.expr[self.pos] == tok:
                    self.pos += 1
                    toklen += 1

            return (self.SINGLE_OR_DOUBLE_CHAR_TOKS[tok][toklen-1], tok+tok)

        if self.expr[self.pos] == '=':
            self.pos += 1
            if self.pos < self.length:
                if self.expr[self.pos] == '=':
                    self.pos += 1
                    return (Token.EQ, '==')

            raise TokenizerError(self.pos, "Expected '='.")

        if self.expr[self.pos] == '!':
            self.pos += 1
            if self.pos < self.length:
                if self.expr[self.pos] == '=':
                    self.pos += 1
                    return (Token.NEQ, '!=')
            return (Token.LOGICAL_NOT, '!')

        return None

    def try_lex_needle(self, needle, tok_type):
        """Look for the given string, and tokenize it if present.

        If present, return the token and update pos, otherwise update nothing
        and return None.
        """
        needle_len = len(needle)
        if self.pos + needle_len > self.length:
            return None

        if self.expr[self.pos:self.pos+needle_len] == needle:
            self.pos += len(needle)
            return (tok_type, needle)

        return None


    def try_lex_defined(self):
        """Tokenize the 'defined' token if present.

        If present, return the token and update pos, otherwise update nothing
        and return None.
        """

        return self.try_lex_needle('defined', Token.DEFINED)


    IDENTIFIER_BEGIN_CHARS = '_' + string.ascii_letters
    IDENTIFIER_NEXT_CHARS = '_' + string.ascii_letters + string.digits

    def try_lex_identifier(self):
        """Tokenize an identifier if present.

        If present, return the token and update pos, otherwise update nothing
        and return None.
        """

        if self.expr[self.pos] not in self.IDENTIFIER_BEGIN_CHARS:
            return None

        begin = self.pos
        self.pos += 1

        while self.pos < self.length:
            if self.expr[self.pos] not in self.IDENTIFIER_NEXT_CHARS:
                break
            self.pos += 1

        return (Token.IDENTIFIER, self.expr[begin:self.pos])

    def try_lex_numeric_literal(self):
        """Tokenize a numeric literal if present.

        If present, return the token and update pos, otherwise update nothing
        and return None.
        """

        if self.expr[self.pos] not in string.digits:
            return None

        begin = self.pos
        self.pos += 1
        while self.pos < self.length:
            if self.expr[self.pos] not in string.digits:
                break
            self.pos += 1

        return (Token.NUMERIC_LITERAL, self.expr[begin:self.pos])

    def try_lex_true_false(self):
        """Tokenize true/false if present.

        If present, return the token and update pos, otherwise update nothing
        and return None.
        """

        tok = self.try_lex_needle('true', Token.TRUE)
        if tok is not None:
            return tok

        tok = self.try_lex_needle('false', Token.FALSE)
        if tok is not None:
            return tok

        return None

    def get_next_internal(self):
        """Return the next token. If at the end of the stream, return None.

        Tokens are returned as a pair (type, str) where type is the type of
        token (see the constants above), and str is the actual token as a
        string.
        """
        if self.pos >= self.length:
            return None

        tok = self.try_lex_operator()
        if tok is not None:
            return tok

        tok = self.try_lex_defined()
        if tok is not None:
            return tok

        # This has to happen before trying to parse an identifier, because
        # true/false both could be interpreted as identifiers otherwise.
        tok = self.try_lex_true_false()
        if tok is not None:
            return tok

        tok = self.try_lex_identifier()
        if tok is not None:
            return tok

        tok = self.try_lex_numeric_literal()
        if tok is not None:
            return tok

        # If we get here, the token is unknown. Just eat the single character.
        tok = (Token.UNKNOWN, self.expr[self.pos])
        self.pos += 1
        return tok

    def get_next(self):
        """Return the next token. If at the end of the stream, return None.

        Tokens are returned as Token objects.
        """

        # Eat whitespace first. It'll get tacked onto the beginning of
        # whatever token is parsed.
        start = self.pos
        self.consume_whitespace()

        tok = self.get_next_internal()
        if tok is None:
            return None

        return Token(tok[0], self.expr[start:self.pos], start, self.pos)

    def next(self):
        """Present an iterator interface.

        See get_next() for what is returned. The only difference is, this raises
        StopIteration instead of returning None when at the end of the stream.
        """

        tok = self.get_next()
        if tok is None:
            raise StopIteration()

        return tok

class Simplifier:
    """Simplify preprocessor conditionals as much as possible."""

    def __init__(self, defines, undefines):
        self.defines = build_defines_map(defines)
        self.undefines = set(undefines)

    def try_convert_to_boolean(self, expr):
        """Cast a value to a boolean if possible.

        If the given expression is not simplified down to a single value, return
        the expression itself. If it is, return True or False.
        """

        if expr == 'true':
            return True
        if expr == 'false':
            return False

        try:
            return int(expr) != 0
        except ValueError:
            return expr

    def simplify(self, expr):
        """Simplify the given expression, substituting in all defines/undefines.

        If the expression can be shown to be true or false, given the set of
        defines and undefines, return True or False. Otherwise, simplify the
        expression as much as possible, and return it as a string.
        """

        self.tokenizer = Tokenizer(expr)
        self.expr = expr
        self.cur_tok = self.tokenizer.get_next()
        return self.try_convert_to_boolean(self.simplify_expr())

    def get_next_tok(self):
        """Consume the current token and return the next one."""
        self.cur_tok = self.tokenizer.get_next()
        return self.cur_tok

    def expect_and_consume(self, kind):
        """Consume and return the given token kind.
        
        If the token kind is not present, raise an error.
        """

        tok = self.cur_tok
        self.get_next_tok()

        if tok.kind != kind:
            raise TokenizerError(tok.start,
                                 'Expected token kind {}.'.format(kind))

        return tok

    def simplify_expr(self):
        """Simplify a full expression using operator precedence parsing.

        This being parsing the expression by parsing the leftmost
        primary expression.
        """
        
        return self.simplify_expr_op(self.simplify_primary(), 0)

    def simplify_binop(self, op, lhs, rhs):
        """Try to simplify the expression lhs op rhs."""

        lhs = self.try_convert_to_boolean(lhs)
        rhs = self.try_convert_to_boolean(rhs)
        lhs_bool = isinstance(lhs, bool)
        rhs_bool = isinstance(rhs, bool)
        both_bool = lhs_bool and rhs_bool

        # If exactly one argument's value is known, make it so that one is lhs,
        # to simplify the code
        if rhs_bool and not lhs_bool:
            (rhs, lhs) = (lhs, rhs)
            (rhs_bool, lhs_bool) = (lhs_bool, rhs_bool)

        if op.kind == Token.LOGICAL_AND:
            if both_bool:
                return lhs and rhs

            if lhs_bool:
                if lhs_val:
                    return lhs.get_leading_whitespace() + rhs
                else:
                    return op.get_leading_whitespace() + 'false'

        if op.kind == Token.LOGICAL_OR:
            if both_bool:
                return lhs or rhs

            if lhs_bool:
                if lhs_val:
                    return op.get_leading_whitespace() + 'true'
                else:
                    return rhs


        # We couldn't simplify anything.
        return lhs + ' ' + op + ' ' + rhs


    def simplify_expr_op(self, lhs, min_precedence):
        """Perform the actual operator precedence parsing and simplification.
        """

        while self.cur_tok is not None:
            if not self.cur_tok.is_binary_operator():
                break;
            if self.cur_tok.get_operator_precedence() < min_precedence:
                break

            op = self.cur_tok
            op_prec = op.get_operator_precedence()
            self.get_next_tok()
            rhs = self.simplify_primary()

            while self.cur_tok is not None:
                if self.cur_tok.get_operator_precedence() is None:
                    break

                prec = self.cur_tok.get_operator_precedence()
                if self.cur_tok.is_binary_operator() and prec <= op_prec:
                    break

                lookahread = self.cur_tok
                lookahead_prec = lookahead.get_operator_precedence()
                self.get_next_tok()
                rhs = self.simplify_expr_op(rhs, lookahead_prec)

            # Here we try to simplify LHS op RHS and store it in LHS.
            lhs = self.simplify_binop(op, lhs, rhs)

        return lhs

    ALREADY_SIMPLIFIED_KINDS = [Token.NUMERIC_LITERAL,
                                Token.TRUE,
                                Token.FALSE]

    def simplify_primary(self):
        """Simplify a primary expression.

        primary ::= '(' expression ')'
                  | IDENTIFIER
                  | NUMERIC_LITERAL
                  | 'defined' '(' IDENTIFIER ')'
                  | 'defined' IDENTIFIER
        """
        
        if self.cur_tok.kind in self.ALREADY_SIMPLIFIED_KINDS:
            tok = self.cur_tok
            self.get_next_tok()
            return tok.val

        if self.cur_tok.kind == Token.DEFINED:
            first_tok = self.cur_tok
            self.get_next_tok()

            parens = False
            if self.cur_tok.kind == Token.LPAREN:
                parens = True
                self.get_next_tok()

            ident = self.expect_and_consume(Token.IDENTIFIER)
            last_tok = ident

            if parens:
                last_tok = self.expect_and_consume(Token.RPAREN)

            value = None
            pp_def = ident.val.lstrip()
            if pp_def in self.defines:
                value = True
            elif pp_def in self.undefines:
                value = False

            if value is None:
                return self.expr[first_tok.start:last_tok.end+1]

            if value:
                return first_tok.get_leading_whitespace() + '1'
            else:
                return first_tok.get_leading_whitespace() + '0'

        if self.cur_tok.kind == Token.IDENTIFIER:
            tok = self.cur_tok
            self.get_next_tok()

            pp_def = tok.val.lstrip()
            value = None
            if pp_def in self.defines:
                return self.defines[pp_def]

            return tok.val

        tok = self.cur_tok
        self.get_next_tok()
        raise TokenizerError(tok.start, 'Unexpected token kind {}.'.format(tok.kind))

class Parser:
    """Parse and preprocess source code."""

    def __init__(self, source, defines, undefines):
        self.source = source
        self.lines = source.splitlines()
        self.cur_line = 0
        self.defines = build_defines_map(defines)
        self.undefines = set(undefines)

        self.ifdef_matcher = re.compile('^#ifdef (.+)')
        self.endif_matcher = re.compile('^#endif')
        self.if_matcher = re.compile('^#if (.+)')

        self.simplifier = Simplifier(defines, undefines)

    def lookup_define(self, sym):
        """If the given preprocessor symbol is defined, return its value.
        
        Symbols that are defined but not given a value will yield the empty
        string. Return None for symbols that are not defined.
        """
        if sym in self.defines:
            return self.defines[sym]
        else:
            return None

    def is_undefined(self, sym):
        """Return whether the given preprocessor symbol is undefined."""
        return sym in self.undefines

    def is_at_end(self):
        """Return whether all the lines have been parsed."""
        return self.cur_line >= len(self.lines)

    def consume_cur_line(self):
        """Read a single line of input and move onto the next."""
        ret = self.lines[self.cur_line]
        self.cur_line += 1
        return ret

    def get_cur_line(self):
        """Return the current line. Don't move onto the next."""
        return self.lines[self.cur_line]

    def consume_until_match(self, matcher):
        """Consume lines until the regexp is matched, or eof is hit.
        
        Return the match result if it's eventually hit, or None if eof is hit.
        """

        while not self.is_at_end():
            m = matcher.match(self.get_cur_line())
            if m is not None:
                return m

            self.consume_cur_line()

        return None

    def process(self):
        """Process the passed-in source file and return it as a list of lines.
        """

        output = []
        while not self.is_at_end():
            out_lines = self.parse_line()
            if out_lines is not None:
                output.extend(out_lines)

        return output

    def parse_line(self):
        """Return parsed output, as a list of lines.

        Consumes all the input that is used.
        If there is no output, return None.
        """

        if self.is_at_end():
            return None

        m = self.ifdef_matcher.match(self.get_cur_line())
        if m is not None:
            return self.parse_ifdef(m.group(1))

        m = self.if_matcher.match(self.get_cur_line())
        if m is not None:
            return self.parse_if(m.group(1))

        return [self.consume_cur_line()]

    def parse_ifdef(self, sym):
        """Parse an #ifdef statement, where sym is the preprocessor symbol.

        If it's not the list of defines, the enclosed code is returned. If it
        is, None is returned. This must be called with the #ifdef as the current
        line, and will parse up to and including the #endif.
        """
        
        # Eat the ifdef
        ifdef_line = self.cur_line
        self.consume_cur_line()

        self.consume_until_match(self.endif_matcher)
        assert not self.is_at_end()

        # Eat the endif
        endif_line = self.cur_line
        self.consume_cur_line()
    
        if self.lookup_define(sym) is not None:
            # If the symbol is defined, then keep the enclosed lines, but get
            # rid of the ifdef.
            first_line = ifdef_line + 1
            end_line = endif_line
        elif self.is_undefined(sym):
            # If the symbol is undefined, remove the whole block.
            return None
        else:
            first_line = ifdef_line
            end_line = endif_line + 1

        return self.lines[first_line:end_line]

    def parse_if(self, expr):
        """Parse an #if statement, where expr is the conditional.

        Tries to simplify the expression using the defined/undefined symbols. If
        the expression can be fully simplified, the #if/#endif will be removed,
        and the source will remain only if the condition is true. Otherwise, the
        simplified expression will be emitted in place of the original one.
        """

        # Eat the if
        if_line = self.cur_line
        self.consume_cur_line()

        self.consume_until_match(self.endif_matcher)
        assert not self.is_at_end()

        # Eat the endif
        endif_line = self.cur_line
        self.consume_cur_line()

        simplified = self.simplifier.simplify(expr)

        # If the condition is known to be true, then keep the enclosed
        # lines, but get rid of the if.
        if simplified is True:
            first_line = if_line + 1
            end_line = endif_line
            return self.lines[first_line:end_line]

        # If the condition is known to be false, remove the whole block.
        if simplified is False:
            return None

        # Otherwise, the condition's value isn't known, but it might have been
        # simplified. Replace the expression with the simplified one.
        new_line = self.lines[if_line].replace(expr, simplified)
        return [new_line] + self.lines[if_line+1:endif_line+1]

def process(source, defines, undefines):
    """Preprocess source, only using definitions from defines.

    defines is a list, where each element is either a string, indicating
    that preprocessor symbol should be defined, or a tuple of two strings,
    indicating that the former is defined to be the latter.
    """

    parser = Parser(source, defines, undefines)
    output_lines = parser.process()
    
    return '\n'.join(output_lines)

class InvalidCommandLineArgError(Exception):
    def __init__(self, arg):
        self.arg = arg
    def __str__(self):
        return self.arg

def parse_args(args):
    """Extract preprocessor definitions from the command line.

    Each element of args should either be of the form -DFOO or -DFOO=BAR for
    defines, and -UFOO for undefines.
    The return value is (defines, undefines).
    The defines are a list, where each element os either a string, or a tuple
    of two strings. The undefines are a list of strings.
    """

    eq_matcher = re.compile('^\-D(.+)$')
    defn_matcher = re.compile('^\-D(.+)=(.+)$')
    undef_matcher = re.compile('^\-U(.+)$')

    defines = []
    undefines = []

    for arg in args:
        m = defn_matcher.match(arg)
        if m is not None:
            defines.append((m.group(1), m.group(2)))
            continue
        
        m = eq_matcher.match(arg)
        if m is not None:
            defines.append(m.group(1))
            continue

        m = undef_matcher.match(arg)
        if m is not None:
            if '=' not in m.group(1):
                undefines.append(m.group(1))
                continue

        raise InvalidCommandLineArgError(arg)

    return (defines, undefines)
