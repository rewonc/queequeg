#!/bin/bash
set -e

echo "Syncing to pogo"
rsync -r code rewon@pogo:~/right_whale/
echo "Downloading any reports"
rsync -r rewon@pogo:~/right_whale/reports .
