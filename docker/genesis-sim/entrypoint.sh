#!/bin/bash
# Start virtual X display for headless Genesis rendering
Xvfb :99 -screen 0 1920x1080x24 &
sleep 1

# Run the provided command
exec "$@"
