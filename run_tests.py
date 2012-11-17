"""Run all the file comparison tests"""

import difflib
import os
import os.path
import partial_pp
import re
import sys

def get_test_inputs(tests_dir):
    """Return the list of input files for testing in tests_dir"""
    
    match_inputs = re.compile('.+\\.c$')
    return [f for f in os.listdir(tests_dir) if match_inputs.match(f)]

def get_defines_and_undefines_from_input(input):
    """Takes the input file and extracts the preprocessor definitions that it
    should be tested with.
    """

    first_line = input.split('\n', 1)[0]
    cmd_args = re.match('// RUN: (.+)$', first_line).group(1).split(' ')
    return partial_pp.parse_args(cmd_args)

def run_test(test_path):
    """Runs the preprocessor on the file and checks for correctness."""

    with open(test_path, 'r') as input_file:
        input = input_file.read().rstrip()

    with open(test_path+'.out', 'r') as output_file:
        reference = output_file.read().rstrip()

    (defines, undefines) = get_defines_and_undefines_from_input(input)
    output = partial_pp.process(input, defines, undefines)

    if reference != output:
        print('[!!!] {} failed:'.format(test_path))
        print('[!!!] Input:')
        print(input)
        print('[!!!] Output:')
        print(output)
        print('[!!!] Diff:')
        diff = difflib.ndiff(reference.splitlines(1), output.splitlines(1))
        print(''.join(diff))
    else:
        print('[===] {} succeeded.'.format(test_path))

def main(argv):

    test_dir = os.path.join(os.getcwd(), 'tests')
    for file in get_test_inputs(test_dir):
        run_test(os.path.join('tests/', file))

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
