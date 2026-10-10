#!/usr/bin/env python3
"""Fresh run without cached imports."""
import subprocess
import sys
sys.exit(subprocess.run([sys.executable, "/home/ubuntu/RAGGA/run_ingestion.py"]).returncode)
