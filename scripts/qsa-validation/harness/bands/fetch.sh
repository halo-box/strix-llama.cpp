#!/bin/bash
# fetch ~/dev/wt_lab/bands/{res,logs,*.sh,*.cu,*.h,*.cpp,*.diff} from strix-002 (the login banner pollutes stdout, so base64 between markers)
cd "$(dirname "$0")/../remote"
ssh strix-002 'cd ~/dev/wt_lab/bands && echo __BEGIN__ && tar czf - res logs campaign.sh bands_ref.h bands_attn.cu bands_dec.cu bands_cpu.cpp tree_bands.diff | base64 -w0 && echo && echo __END__' 2>/dev/null | sed -n '/__BEGIN__/,/__END__/p' | sed '1d;$d' | base64 -d | tar xzf -
