# Bug fixtures

Minimal inputs for the tests labelled `bug` (`ctest -L bug`). Each of those tests checks a documented
contract that the code does not meet yet, so each one fails until its issue is fixed. The tests name the issue
in a Catch2 tag, `[issue-N]`.

| Directory | Format | Tests |
|---|---|---|
| `aiger/` | AIGER ASCII (`aag`), stored byte for byte (CRLF) | `src/io/aiger/tests/bug_tests.cc` |
| `ftp/` | CAFTA FTP projects | `src/compile/ftp/tests/bug_tests.cc` |
| `nombrilo/` | nombrilo.circuit safetensors, malformed on purpose | `src/compile/tests/nombrilo_bug_tests.cc` |
| `openpsa/` | OpenPSA MEF XML, each valid against `share/input.rng` | `src/compile/tests/mef_bug_tests.cc` |

Each OpenPSA file starts with a comment giving its expected value and how it follows from the model. Some
files have no test: there the expected behaviour is a decision still to be made, and the file's issue says
which. `openpsa/atleast_15_of_30.xml` does not finish with `fzb compile --no-kn`.

The fixtures sit apart from the format directories (`input/aiger`, `input/ftp/...`), so corpus-wide tests
never read them. When a bug is fixed, its test leaves the `bug` label and becomes an ordinary test. Its
fixture moves next to the other inputs of its format.
