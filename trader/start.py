"""
Запуск Investor Terminal.
Использование: python start.py [--port 8000]
"""
import sys
import subprocess
from pathlib import Path

PORT = 8000
for i, arg in enumerate(sys.argv):
    if arg == "--port" and i+1 < len(sys.argv):
        PORT = int(sys.argv[i+1])

ROOT = Path(__file__).parent
print(f"\n{'='*50}")
print("  Investor Terminal v1.0")
print(f"  API:       http://localhost:{PORT}/api/health")
print(f"  Docs:      http://localhost:{PORT}/docs")
print(f"  Дашборд:   {ROOT / 'frontend' / 'index.html'}")
print("  Тесты:     python tests/test_all.py")
print(f"{'='*50}\n")

subprocess.run([
    sys.executable, "-m", "uvicorn", "api.main:app",
    "--host", "0.0.0.0", "--port", str(PORT), "--reload"
], cwd=str(ROOT))
