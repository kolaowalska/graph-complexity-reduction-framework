import sys
from src.interfaces.cli import run_cli


def main(args=None):
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
