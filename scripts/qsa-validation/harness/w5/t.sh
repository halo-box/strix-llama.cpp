#!/bin/bash
# W5: one test-backend-ops run under flock (one op, -j 1 default); usage: t.sh <tag> <env...> <cmd...>
cd ~/dev/wt_rewrite; L=~/dev/wt_rewrite_logs/W5; tag=$1; shift
flock ~/gpu.lock env "$@" > $L/$tag.log 2>&1; rc=$?
echo "EXIT $tag rc=$rc $(grep -h "tests passed" $L/$tag.log | tr -s " ") fail=$(grep -v "SA_REPORT\|SA_GQAREP" $L/$tag.log | grep -c "FAIL") notsup=$(grep -c "not supported" $L/$tag.log) $(grep -h "CENSUS COVERAGE: [A-Za-z0-9]*: [0-9]" $L/$tag.log | tr -s " ")"
