#!/bin/bash
set -e

echo "Syncing to pogo"
rsync -r code rewon@pogo:~/right_whale/
