#!/bin/bash
# run a command on strix-002 with the login banner filtered
ssh strix-002 "$@" 2>&1 | grep -v -E 'HOMEBREW|^export (PATH|INFOPATH)|MANPATH'
