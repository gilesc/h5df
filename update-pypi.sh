#!/usr/bin/env bash

if [[ "$1" == "test" ]]; then
    repo="pypitest"
elif [[ "$1" == "release" ]]; then
    repo="pypi"
else
    echo "USAGE: $0 [test|release]" 1>&2
    exit 1
fi

python setup.py sdist upload -r $repo
