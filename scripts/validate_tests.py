"""Validate test file structure and imports.

This script checks that all test files are properly structured and can be imported.
Run before committing to catch import errors early.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_test_file(test_file: Path) -> tuple[bool, str]:
    """Validate a single test file.

    Args:
        test_file: Path to test file

    Returns:
        Tuple of (success, message)
    """
    try:
        # Try to import the test module
        module_name = f"tests.{test_file.stem}"
        __import__(module_name)
        return True, f"✓ {test_file.name}"
    except Exception as e:
        return False, f"✗ {test_file.name}: {str(e)}"


def main():
    """Validate all test files."""
    tests_dir = project_root / "tests"

    if not tests_dir.exists():
        print("Error: tests directory not found")
        sys.exit(1)

    # Find all test files
    test_files = sorted(tests_dir.glob("test_*.py"))

    if not test_files:
        print("Warning: No test files found")
        sys.exit(0)

    print(f"Validating {len(test_files)} test files...\n")

    results = []
    for test_file in test_files:
        success, message = validate_test_file(test_file)
        results.append(success)
        print(message)

    print(f"\n{'='*50}")
    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")

    if failed > 0:
        print("\n⚠ Some test files have import errors. Fix them before running pytest.")
        sys.exit(1)
    else:
        print("\n✓ All test files validated successfully!")
        print("\nYou can now run: pytest tests/")
        sys.exit(0)


if __name__ == "__main__":
    main()
