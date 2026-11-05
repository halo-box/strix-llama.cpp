#!/bin/bash
ssh strix-002 "$@" 2>&1 | grep -vE '^(eval )?export |MANPATH|^export INFOPATH|HOMEBREW'
