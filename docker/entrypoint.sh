#!/bin/sh
set -eu

mkdir -p /home/mediko/.paddlex
chown -R mediko:mediko /home/mediko/.paddlex

exec runuser -u mediko -- "$@"
