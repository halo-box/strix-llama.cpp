#!/usr/bin/env python3
# C7 lab sampler: every PERIOD s, CLOCK_MONOTONIC ns + KFD evicted_ms of the pid + meminfo + vmstat + process faults.
import glob, os, sys, time

pid = int(sys.argv[1]); out = open(sys.argv[2], "w"); period = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
VM = ["pswpin", "pswpout", "pgmajfault", "allocstall_normal", "allocstall_movable", "compact_stall", "pgscan_direct",
      "pgsteal_direct", "pgscan_kswapd", "thp_fault_alloc", "numa_pages_migrated", "pgmigrate_success"]
MI = ["MemFree", "MemAvailable", "SwapFree", "Cached"]
out.write("t_ns evicted_ms cu_occ " + " ".join(MI) + " " + " ".join(VM) + " minflt majflt\n")
def rd(p):
    try:
        with open(p) as f: return f.read()
    except OSError: return None
while os.path.exists(f"/proc/{pid}"):
    t = time.monotonic_ns()
    ev = "-1"; occ = "-1"
    for d in glob.glob(f"/sys/class/kfd/kfd/proc/{pid}/stats_*"):
        e = rd(d + "/evicted_ms"); c = rd(d + "/cu_occupancy")
        if e is not None: ev = e.strip()
        if c is not None: occ = c.strip()
    mi = {}; s = rd("/proc/meminfo") or ""
    for ln in s.splitlines():
        k, _, v = ln.partition(":"); mi[k] = v.split()[0] if v.split() else "0"
    vm = {}; s = rd("/proc/vmstat") or ""
    for ln in s.splitlines():
        k, _, v = ln.partition(" "); vm[k] = v
    st = rd(f"/proc/{pid}/stat")
    if st is None: break
    f = st.rsplit(")", 1)[1].split()
    out.write(f"{t} {ev} {occ} " + " ".join(mi.get(k, "0") for k in MI) + " " + " ".join(vm.get(k, "0") for k in VM) + f" {f[7]} {f[9]}\n")
    time.sleep(period)
out.close()
