#!/bin/bash
# W3 final measurements on the final kernel: (1) kernel-time campaign, (2) G4 suite with report + TBO dumps, the
# PR-verbatim kernel on those dumps, and the comparison; every GPU step under flock ~/gpu.lock (inside the scripts)
cd ~/dev/wt_lab/w3/c18 && rm -rf res_v1 && mv res res_v1 && mv summary.md summary_v1.md && ./campaign.sh > campaign2.out 2>&1 && python3 summ.py res > summary.md 2>&1; echo "CAMPAIGN $(grep -c EXIT campaign2.out) runs, $(grep 'GATE' summary.md)"
rm -rf ~/dev/wt_lab/w3/dump2; mkdir -p ~/dev/wt_lab/w3/dump2; cd ~/dev/wt_rewrite
flock ~/gpu.lock env GGML_SA_TEST_REPORT=1 GGML_SA_TEST_DUMP=$HOME/dev/wt_lab/w3/dump2 build/bin/test-backend-ops test -o SPARSE_ATTN -b ROCm0 -p group=4 --fail-unsupported > ~/dev/wt_rewrite_logs/W3/t5_g4.log 2> ~/dev/wt_rewrite_logs/W3/t5_g4.err
echo "T5 rc=$? $(grep 'tests passed' ~/dev/wt_rewrite_logs/W3/t5_g4.log)"
cd ~/dev/wt_lab/w3/prband && rm -rf res2 && ./run.sh ~/dev/wt_rewrite_logs/W3/t5_g4.err res2 > run2.out 2>&1 && python3 an.py run2.out res2 ~/dev/wt_rewrite_logs/W3/t5_g4.log > summary2.md; tail -3 summary2.md
echo DONE
