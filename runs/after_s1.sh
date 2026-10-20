#!/usr/bin/env bash
# Wait for the seed-1 queue, then start the dose jobs.
cd "$(dirname "$0")/.."
while pgrep -f "^bash ./queue.sh" >/dev/null; do sleep 30; done
JOBS="dose100_s0 dose250_s0 dose1000_s0 dose2000_s0" make start
