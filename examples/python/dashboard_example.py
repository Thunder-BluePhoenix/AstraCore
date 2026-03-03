"""
Dashboard from Python — AstraCore
===================================
All three dashboard backends (HTML report, browser server, TUI) are
accessible directly from Python via the `astracore` module.

Dashboard API
-------------
circuit.html_report("report.html")     # write standalone HTML file
circuit.serve(port=8080)               # blocking HTTP server (Ctrl-C to stop)
circuit.dash()                         # blocking terminal TUI (q to quit)

Free-function equivalents (no Circuit object needed):
    run_aql_report(source, "report.html")
    run_aql_serve(source, port=8080)
    run_aql_dash(source)

Run:
    python examples/python/dashboard_example.py
"""
import sys
import os
from astracore import Circuit, run_aql_report, run_aql_serve, run_aql_dash

# ── Circuit: 3-qubit GHZ state ────────────────────────────────────────────────

c = Circuit(3)
c.h(0)
c.cnot(0, 1)
c.cnot(0, 2)
c.measure_all()

# ── 1. HTML Report ────────────────────────────────────────────────────────────
# Generates a standalone report.html — open in any browser.
# No server required; Chart.js is embedded.

report_path = "ghz_report.html"
c.html_report(report_path)
print(f"HTML report written → {os.path.abspath(report_path)}")
print(f"  Open in browser: file://{os.path.abspath(report_path)}")
print()

# ── 2. Free function variant ──────────────────────────────────────────────────

bell_source = "QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL"
run_aql_report(bell_source, "bell_report.html")
print(f"Bell report written → {os.path.abspath('bell_report.html')}")
print()

# ── 3. Browser dashboard or TUI (choose via argv) ─────────────────────────────

mode = sys.argv[1] if len(sys.argv) > 1 else "none"

if mode == "serve":
    # Launch interactive browser dashboard.
    # Open http://localhost:8080 in your browser.
    # The SPA editor lets you modify AQL and re-run circuits live.
    # Press Ctrl-C to stop.
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    print(f"Starting browser dashboard at http://localhost:{port}")
    print("Press Ctrl-C to stop.")
    c.serve(port=port)

elif mode == "dash":
    # Launch the terminal TUI dashboard.
    # Requires a real TTY (won't work in Jupyter).
    # Press q or Esc to quit.
    print("Launching TUI dashboard (press q to quit)...")
    c.dash()

elif mode == "serve-aql":
    # Same as serve but from raw AQL source
    source = """
QREG 4
H 0
CNOT 0 1
CNOT 0 2
CNOT 0 3
MEASURE_ALL
"""
    print("Starting 4-qubit GHZ browser dashboard at http://localhost:8080")
    run_aql_serve(source, port=8080)

else:
    print("Usage:")
    print("  python dashboard_example.py           — generate HTML reports only")
    print("  python dashboard_example.py serve      — start browser dashboard")
    print("  python dashboard_example.py serve 9090 — use port 9090")
    print("  python dashboard_example.py dash       — terminal TUI")
    print("  python dashboard_example.py serve-aql  — serve 4-qubit GHZ")
