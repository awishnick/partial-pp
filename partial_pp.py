"""Partially preprocess a C/C++ source file, only acting on certain preprocessor
definitions.
"""

import re

class Parser:
    def __init__(self, source, defines, undefines):
        self.source = source
        self.lines = source.splitlines()
        self.cur_line = 0
        self.defines = {}
        self.undefines = set(undefines)

        for define in defines:
            if isinstance(define, tuple):
                (symbol, definition) = define
                self.defines[symbol] = definition
            else:
                assert isinstance(define, str)
                self.defines[define] = ''

        self.ifdef_matcher = re.compile('^#ifdef (.+)')
        self.endif_matcher = re.compile('^#endif')

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

    eq_matcher = re.compile('\-D(.+)')
    defn_matcher = re.compile('\-D(.+)=(.+)')
    undef_matcher = re.compile('\-U(.+)')

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
            undefines.append(m.group(1))
            continue

        raise InvalidCommandLineArgError(arg)

    return (defines, undefines)
