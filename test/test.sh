#!/usr/bin/env bash

matrix=$(dirname $0)/matrix.txt
db=$(mktemp)
hpath="/test"

trap 'rm -rf $db' EXIT SIGTERM SIGKILL

call() {
    python3 $(dirname $0)/../h5df.py "$@"
}

echo "* Load" 1>&2
call load $db $hpath < $matrix

echo -e "\n* Dump" 1>&2
call dump $db $hpath

echo -e "\n* Row" 1>&2
call row $db $hpath X

echo -e "\n* Column" 1>&2
call column $db $hpath A

echo -e "\n* Rows" 1>&2
echo -e "X\nY" | call rows $db $hpath

echo -e "\n* Columns" 1>&2
echo -e "A\nC" | call columns $db $hpath
