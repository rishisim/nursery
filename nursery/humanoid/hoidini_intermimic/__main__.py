"""Convert a saved HOIDiNi run to the upstream InterMimic reference format."""

import argparse

from .convert import convert_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Step-2 run containing motion, input, object, and validation")
    parser.add_argument("--humanoid-xml", required=True, help="Explicit upstream 52-body SMPL-X MJCF")
    parser.add_argument("--object-name", required=True, help="Single object key used by InterMimic's filename parser")
    parser.add_argument("--output", required=True, help="New ignored output directory")
    args = parser.parse_args()
    convert_run(args.run, args.humanoid_xml, args.output, args.object_name)


if __name__ == "__main__":
    main()
