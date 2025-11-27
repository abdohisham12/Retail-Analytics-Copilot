#!/usr/bin/env python
"""Test script to verify output is visible"""
import sys
import time

print("Starting test...", flush=True)
sys.stdout.flush()
sys.stderr.flush()

for i in range(5):
    print(f"Progress: {i+1}/5", flush=True)
    sys.stdout.flush()
    time.sleep(0.5)

print("Test complete!", flush=True)
sys.stderr.write("Stderr test message\n")
sys.stderr.flush()

