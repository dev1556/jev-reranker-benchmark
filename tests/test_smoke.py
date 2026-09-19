import sys


def test_python_version_is_pinned() -> None:
    """torch has no 3.13+ wheels; arm B dies if this drifts."""
    assert sys.version_info[:2] == (3, 12)


def test_src_package_imports() -> None:
    import src

    assert src is not None
