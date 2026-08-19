from __future__ import annotations

import argparse
import json
from pathlib import Path

from manga_hd_transfer.direct_patch_bridge import dispatch_isolated_direct_mode


def main() -> None:
    ap = argparse.ArgumentParser(description='Run isolated direct borderless overlay mode on a Folirina page workspace.')
    ap.add_argument('page_dir', help='Path to page folder, e.g. pages/p-044')
    ap.add_argument('--report', default='', help='Optional JSON report output path')
    args = ap.parse_args()

    payload = dispatch_isolated_direct_mode('direct_patch', args.page_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
