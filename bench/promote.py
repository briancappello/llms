#!/usr/bin/env python3
"""Make a measured benchmark variant the production configuration.

A benchmark conclusion is finished when production serves the winner. Name the
results file and the winning variant; the patch is taken from the file, not
retyped. Production must still be the configuration the variant was measured
against (base digest and rendered command), or nothing changes.

    bench/promote.py results/speed/cyber-tiel-kv.tsv --variant q8V

On success the service restarts on the new config, the model is warmed, and a
row is appended to results/promotions.tsv. On a failed restart or warm-up the
previous files are restored and the service restarted on them.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from promote import PromoteError, promote  # noqa: E402
from stack import Settings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", metavar="RESULTS", help="a stack-mode results TSV or kv-probe JSON")
    parser.add_argument("--variant", required=True, help="the variant to promote, as named in RESULTS")
    parser.add_argument("--model", help="the measured model, when RESULTS covers several")
    parser.add_argument("--config-dir", help="production llms config dir (default: the host's)")
    parser.add_argument("--accept-shared-engine", action="store_true",
                        help="also apply an engine change to the other models that use that engine")
    args = parser.parse_args()
    try:
        production = Settings.from_env(args.config_dir)
        warmup = promote(args.results, args.variant, model=args.model, production=production,
                         accept_shared_engine=args.accept_shared_engine)
    except (PromoteError, ValueError, OSError) as error:
        print(f"promote: {error}", file=sys.stderr)
        return 1
    print(f"promoted {args.variant} from {args.results} (warm-up: {warmup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
