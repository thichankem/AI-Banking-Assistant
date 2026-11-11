"""Entry point for chunking the banking corpus."""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.chunking.run_chunking import main, run_auto_chunking


if __name__ == "__main__":
    main(default_profile="accuracy")
