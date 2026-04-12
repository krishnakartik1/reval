"""Zero-dep guard for `reval.contracts`.

The contracts namespace is meant to be portable across reval and
reval-factual-collector. To keep it transitively dependency-light, this
test imports `reval.contracts` in a fresh Python subprocess and asserts
that none of the forbidden modules got pulled in.

Running in a subprocess is important because pytest itself imports
`reval`, which may have already loaded many of these modules into
`sys.modules` by the time the test runs — an in-process `sys.modules`
check would give false negatives.
"""

import subprocess
import sys
import textwrap


FORBIDDEN = ["aioboto3", "boto3", "numpy", "jsonlines", "httpx", "anthropic", "openai"]


def test_contracts_has_no_forbidden_transitive_imports() -> None:
    program = textwrap.dedent(
        f"""
        import sys
        import reval.contracts  # noqa: F401
        forbidden = {FORBIDDEN!r}
        loaded = [m for m in forbidden if m in sys.modules]
        if loaded:
            print("LOADED:" + ",".join(loaded))
            sys.exit(1)
        sys.exit(0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "reval.contracts must not transitively import heavy deps; "
        f"got {result.stdout.strip()} {result.stderr.strip()}"
    )
