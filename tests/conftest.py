"""Skip tests/ui/ when Playwright is not installed."""

try:
    import playwright.sync_api  # noqa: F401
except ImportError:
    collect_ignore = ["ui"]
