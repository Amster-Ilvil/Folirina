from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from manga_hd_transfer.gui_qt import main
main()
