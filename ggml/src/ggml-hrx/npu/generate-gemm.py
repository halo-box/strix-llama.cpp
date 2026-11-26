"""Generate the XDNA2 signed INT8 GEMM schedule with direct output DMA."""
import argparse
from pathlib import Path

COLS = 8
REC = 8192          # bytes of one A/B record (64 x 128 s8) and of one C half (32 x 64 s32)
CREC = REC // 4
OWNERS = (2, 4)


class Writer:
    def __init__(self):
        self.lines = []
        self.n = 0

    def __call__(self, s):
        self.lines.append(s)

    def t(self, base="t"):
        self.n += 1
        return f"%{base}{self.n}"


def gen(K, G):
    ring = 5
    eslot = 1
    assert K % 256 == 0 and K >= 512, "K must be a multiple of 256, at least 512"
    assert G in (1, 2)
    KC = K // 128
    CPG = COLS // G
    opt = dict(KC=KC, G=G, CPG=CPG, ring=ring, eslot=eslot)
    w = Writer()
    I8, V8 = f"tile<{REC}xi8>", f"view<{REC}xi8>"
    I32, V32 = f"tile<{4 * CREC}xi32>", f"view<{4 * CREC}xi32>"
    J, JV = "tile<1xi32>", "view<1xi32>"
    DMA_IN = "{cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup}"
    DMA_OUT = "{cache_scope = workgroup, cache_temporal = regular, direction = workgroup_to_global}"
    ty = dict(I8=I8, V8=V8, I32=I32, V32=V32, J=J, JV=JV, DMA_OUT=DMA_OUT)

    w("aie2p.target<core> @core")
    w(Path(__file__).with_name("gemm-i8-core.loom.inc").read_text().rstrip("\n"))
    w("config.decl @gemm.tsteps : %value: index where [range(%value, 1, 1024)]")
    w("config.decl @gemm.passes : %value: index where [range(%value, 1, 64)]")
    w("aie2p.target<array> @npu")
    w("pipeline.def<kernel> public retain target(@npu) @gemm_i8() run(%xarg: buffer, %warg: buffer, %yarg: buffer) {")
    w("  %xu, %wu, %yu = buffer.assume.noalias %xarg, %warg, %yarg : buffer, buffer, buffer")
    w("  %x, %wt, %y = buffer.assume.alignment %xu, %wu, %yu {minimum_alignment = 64} : buffer, buffer, buffer")
    for v in (0, 1, 2, 4, 8, 16):
        w(f"  %c{v} = index.constant {v} : index")
    w(f"  %kc = index.constant {KC} : index")
    w(f"  %kcm1 = index.constant {KC - 1} : index")
    w(f"  %cring = index.constant {ring} : index")
    w(f"  %cpg2 = index.constant {2 * CPG} : index")
    w("  %origin = index.constant 0 : offset")
    w("  %w8 = index.constant 8 : offset")
    w("  %w64 = index.constant 64 : offset")
    w(f"  %rec = index.constant {REC} : offset")
    w(f"  %rec2 = index.constant {2 * REC} : offset")
    w(f"  %rec4 = index.constant {4 * REC} : offset")
    w(f"  %ringb = index.constant {ring * REC} : offset")
    w(f"  %aret_bytes = index.constant {KC * REC} : offset")
    w("  %w16k = index.constant 16384 : offset")
    w("  %w56k = index.constant 57344 : offset")
    w("  %tsteps = config.get @gemm.tsteps : index")
    w("  %passes = config.get @gemm.passes : index")
    # 128-column steps of each group: group 0 takes the first ceil(T/G)
    if G == 1:
        w("  %tg0 = index.add %tsteps, %c0 : index")
        w("  %j00 = index.constant 0 : index")
    else:
        w("  %tsp1 = index.add %tsteps, %c1 : index")
        w("  %tg0 = index.div %tsp1, %c2 : index")
        w("  %tg1 = index.sub %tsteps, %tg0 : index")
        w("  %j00 = index.constant 0 : index")
        w("  %j01 = index.add %tg0, %c0 : index")

    # ---------------- storage ----------------
    for c in range(COLS):
        w(f"  %mt{c} = pipeline.memory<workgroup>[{c}, 1] : pool")
        for h in range(2):
            w(f"  %aretb{c}_{h} = buffer.alloca<workgroup> pool(%mt{c}) align(64) %aret_bytes : buffer")
            w(f"  %aretv{c}_{h} = buffer.view %aretb{c}_{h}[%origin] : buffer -> view<{KC}x{REC}xi8>")
            w(f"  %aret{c}_{h} = channel.bind<fifo> %aretv{c}_{h}, %kc : view<{KC}x{REC}xi8>, index -> channel<{I8}>")
        for r in range(2, 6):
            n = f"{c}_{r}"
            w(f"  %core{n} = pipeline.memory<workgroup>[{c}, {r}] : pool")
            w(f"  %bbuf{n} = buffer.alloca<workgroup> pool(%core{n}) align(64) %rec2 : buffer")
            w(f"  %bv{n} = buffer.view %bbuf{n}[%origin] : buffer -> view<2x{REC}xi8>")
            w(f"  %bch{n} = channel.bind<fifo> %bv{n}, %c2 : view<2x{REC}xi8>, index -> channel<{I8}>")
            if r in OWNERS:
                w(f"  %abuf{n} = buffer.alloca<workgroup> pool(%core{n}) align(64) %rec2 : buffer")
                w(f"  %av{n} = buffer.view %abuf{n}[%origin] : buffer -> view<2x{REC}xi8>")
                w(f"  %ach{n} = channel.bind<fifo> %av{n}, %c2 : view<2x{REC}xi8>, index -> channel<{I8}>")
                w(f"  %scrb{n} = buffer.alloca<workgroup> pool(%core{n}) align(64) %w64 : buffer")
                w(f"  %scr{n} = buffer.view %scrb{n}[%origin] : buffer -> view<16xi32>")
            else:
                for nm in ("job", "ack", "cgo", "odone"):
                    w(f"  %{nm}b{n} = buffer.alloca<workgroup> pool(%core{n}) align(64) %w8 : buffer")
                    w(f"  %{nm}v{n} = buffer.view %{nm}b{n}[%origin] : buffer -> view<2x1xi32>")
                    w(f"  %{nm}{n} = channel.bind<fifo> %{nm}v{n}, %c2 : view<2x1xi32>, index -> channel<{J}>")
                for buf, ch, pool in ((f"ringbuf{n}", f"ring{n}", f"core{n}"),):
                    w(f"  %{buf} = buffer.alloca<workgroup> pool(%{pool}) align(64) %rec4 : buffer")
                    w(f"  %{ch}v = buffer.view %{buf}[%origin] : buffer -> view<1x{4 * CREC}xi32>")
                    w(f"  %{ch} = channel.bind<fifo> %{ch}v, %c1 : view<1x{4 * CREC}xi32>, index -> channel<{I32}>")
    for g in range(G):
        for s in range(2):
            st = 2 * g + s
            w(f"  %bstgb{st} = buffer.alloca<workgroup> pool(%mt{st}) align(64) %rec2 : buffer")
            w(f"  %bstgv{st} = buffer.view %bstgb{st}[%origin] : buffer -> view<2x{REC}xi8>")
            w(f"  %bstg{st} = channel.bind<fifo> %bstgv{st}, %c2 : view<2x{REC}xi8>, index -> channel<{I8}>")

    w("  pipeline.serial {")
    w("  scf.for %p = [%c0 to %passes step %c1] {")
    w(f"  %prow = index.mul %p, %cpg2 : index")   # first 64-row block of this pass
    # ---------------- phase 1: A into the retained rings ----------------
    w("  pipeline.concurrent {")
    for c in range(COLS):
        rbl = c % CPG
        for h in range(2):
            ow = OWNERS[h]
            w(f"  pipeline.strand workers([{c}, {ow}], [1, 1], [1, 1]) {{")
            w(f"    %blk = index.constant {2 * rbl + h} : index")
            w("    %rb = index.add %prow, %blk : index")
            w("    %base = index.mul %rb, %kc : index")
            w("    scf.for %r = [%c0 to %kc step %c1] {")
            w("      %ri = index.add %base, %r : index")
            w("      %off = index.scale %ri, %rec : index, offset -> offset")
            w(f"      %ext = buffer.view %x[%off] : buffer -> {V8}")
            w(f"      %wr, %dst = channel.reserve %aret{c}_{h} : channel<{I8}> -> (write<{I8}>, {V8})")
            w(f"      %cp = kernel.async.copy %ext to %dst {DMA_IN} : {V8} to {V8} -> kernel.async.token")
            w("      %g = kernel.async.group %cp : kernel.async.token -> kernel.async.group")
            w("      kernel.async.wait %g {newer_groups = 0} : kernel.async.group")
            w(f"      channel.publish %wr : write<{I8}>")
            w("    }")
            w("  }")
            # instruction worker that starts and retires this ingress
            w(f"  pipeline.strand workers([{c}, {ow}], [1, 1], [1, 1]) {{")
            w("    %zz = vector.splati 0 : vector<16xi32>")
            w(f"    vector.store %zz, %scr{c}_{ow}[0] : vector<16xi32>, view<16xi32>")
            w("  }")
    w("  }")
    # ---------------- phase 2 ----------------
    w("  pipeline.concurrent {")
    for c in range(COLS):
        tg = f"%tg{c // CPG}"
        for h in range(2):
            owner = OWNERS[h]
            w(f"  pipeline.strand workers([{c}, {owner}], [1, 1], [1, 1]) {{")
            for i in range(KC):
                w(f"    %h{i}, %hv{i} = channel.acquire %aret{c}_{h} : channel<{I8}> -> (read<{I8}>, {V8})")
            args = ", ".join(f"%k{i} = %h{i} : read<{I8}>" for i in range(KC))
            rtypes = ", ".join(f"read<{I8}>" for _ in range(KC))
            w(f"    {', '.join(f'%e{i}' for i in range(KC))} = scf.for %j = [%c0 to {tg} step %c1]({args}) -> ({rtypes}) {{")
            for i in range(KC):
                w(f"      %q{i}, %o{i} = channel.fanout %k{i} : read<{I8}> -> (read<{I8}>, read<{I8}>)")
                w(f"      %w{i}, %wv{i} = channel.reserve %ach{c}_{owner} : channel<{I8}> -> (write<{I8}>, {V8})")
                w(f"      channel.copy %o{i} -> %w{i} : read<{I8}>, write<{I8}>")
            w(f"      scf.yield {', '.join(f'%q{i}' for i in range(KC))} : {rtypes}")
            w("    }")
            for i in range(KC):
                w(f"    channel.release %e{i} : read<{I8}>")
            w("  }")
    # B: per group and half, ingress into a memory-tile staging ring and multicast
    for g in range(G):
        cols = range(g * CPG, (g + 1) * CPG)
        for s in range(2):
            st = 2 * g + s
            wk = f"[{st}, {3 if s == 0 else 5}]"
            w(f"  pipeline.strand workers({wk}, [1, 1], [1, 1]) {{")
            w(f"    %sb = index.constant {s} : index")
            w(f"    %jb = index.madd %sb, %tsteps, %j0{g} : index")
            w("    %r0 = index.mul %jb, %kc : index")
            w(f"    %brecs = index.mul %tg{g}, %kc : index")
            w("    scf.for %i = [%c0 to %brecs step %c1] {")
            w("      %ri = index.add %r0, %i : index")
            w("      %off = index.scale %ri, %rec : index, offset -> offset")
            w(f"      %ext = buffer.view %wt[%off] : buffer -> {V8}")
            w(f"      %wr, %dst = channel.reserve %bstg{st} : channel<{I8}> -> (write<{I8}>, {V8})")
            w(f"      %cp = kernel.async.copy %ext to %dst {DMA_IN} : {V8} to {V8} -> kernel.async.token")
            w("      %g = kernel.async.group %cp : kernel.async.token -> kernel.async.group")
            w("      kernel.async.wait %g {newer_groups = 0} : kernel.async.group")
            w(f"      channel.publish %wr : write<{I8}>")
            w("    }")
            w("  }")
            dests = [f"%bch{c}_{r}" for c in cols for r in ((2, 4) if s == 0 else (3, 5))]
            w(f"  pipeline.strand workers({wk}, [1, 1], [1, 1]) {{")
            w(f"    %brecs = index.mul %tg{g}, %kc : index")
            w("    scf.for %i = [%c0 to %brecs step %c1] {")
            w(f"      %rd, %rv = channel.acquire %bstg{st} : channel<{I8}> -> (read<{I8}>, {V8})")
            w(f"      {', '.join(f'%f{d}' for d in range(len(dests)))} = channel.fanout %rd : read<{I8}> -> ({', '.join(f'read<{I8}>' for _ in dests)})")
            for d, dst in enumerate(dests):
                w(f"      %dw{d}, %dv{d} = channel.reserve {dst} : channel<{I8}> -> (write<{I8}>, {V8})")
                w(f"      channel.copy %f{d} -> %dw{d} : read<{I8}>, write<{I8}>")
            w("    }")
            w("  }")
    for c in range(COLS):
        for r in range(2, 6):
            compute_strand(w, c, r, opt, ty)
    w("  }")
    w("  }")
    w("  }")
    w("}")
    return "\n".join(w.lines)


def compute_strand(w, c, r, o, ty):
    KC, CPG, ring = o["KC"], o["CPG"], o["ring"]
    I8, V8, I32, V32, J, JV, DMA_OUT = (ty[k] for k in ("I8", "V8", "I32", "V32", "J", "JV", "DMA_OUT"))
    grp = c // CPG
    rbl = c % CPG
    n = f"{c}_{r}"
    owner = r in OWNERS
    on = n if owner else f"{c}_{r - 1}"   # owner (A ring)
    pn = f"{c}_{r + 1}" if owner else n   # peer (sync channels, C ring)
    # C ring record order per tile: 0 owner lo, 1 peer lo, 2 owner hi, 3 peer hi
    lo_i, hi_i = (0, 2) if owner else (1, 3)
    w(f"  pipeline.strand workers([{c}, {r}], [1, 1], [1, 1]) {{")
    w(f"    %rbl = index.constant {2 * rbl + (r - 2) // 2} : index")
    w(f"    %eslot = index.constant {o['eslot']} : index")
    w(f"    %blkbytes = index.constant {4 * REC} : offset")
    w("    %one32 = scalar.constant 1 : i32")
    w(f"    %ringn = index.constant {ring} : index")
    w(f"    %ilo = index.constant {lo_i} : index")
    w(f"    %ihi = index.constant {hi_i} : index")
    w("    %nz128 = index.constant 128 : index")
    w("    %zvec = vector.splati 0 : vector<16xi32>")
    w(f"    %tg = index.add %tg{grp}, %c0 : index")
    w(f"    %astride = index.constant {REC} : offset")
    w(f"    %aoff0 = index.constant 0 : offset")
    w("    %roff0 = index.constant 0 : offset")
    w(f"    %bstride = index.constant {REC} : offset")
    w(f"    %j0 = index.add %j0{grp}, %c0 : index")

    def calls(a_off, c_lo, c_hi, pairs, ind, b_off=None, first=False):
        kernel = "mk0" if first else "mk"
        b_off = b_off or a_off
        for q in pairs:
            cb = c_lo if q < 2 else c_hi
            qq = q % 2
            a0, a1, pa0o, pa1o, c0, c1, c0o, c1o = (w.t("o") for _ in range(8))
            w(f"{ind}{a0} = index.constant {(2 * q) * 1024} : offset")
            w(f"{ind}{a1} = index.constant {(2 * q + 1) * 1024} : offset")
            a_base = w.t("ab")
            w(f"{ind}{a_base} = index.add {a_off}, %aoff0 : offset")
            w(f"{ind}{pa0o} = index.add {a_base}, {a0} : offset")
            w(f"{ind}{pa1o} = index.add {a_base}, {a1} : offset")
            w(f"{ind}{c0} = index.constant {(2 * qq) * 2048} : offset")
            w(f"{ind}{c1} = index.constant {(2 * qq + 1) * 2048} : offset")
            cbb = w.t("cb")
            w(f"{ind}{cbb} = index.add {cb}, %roff0 : offset")
            w(f"{ind}{c0o} = index.add {cbb}, {c0} : offset")
            w(f"{ind}{c1o} = index.add {cbb}, {c1} : offset")
            ab_ = f"%abuf{on}"
            bb_ = f"%bbuf{n}"
            rb_ = f"%ringbuf{pn}"
            w(f"{ind}low.invoke @{kernel}({ab_}, {pa0o}, {pa1o}, {bb_}, {b_off}, {rb_}, {c0o}, {c1o}) : (buffer, offset, offset, buffer, offset, buffer, offset, offset) -> ()")

    def sig_send(ch, ind):
        sw = w.t("sw")
        w(f"{ind}{sw}, {sw}v = channel.reserve {ch} : channel<{J}> -> (write<{J}>, {JV})")
        w(f"{ind}view.store %one32, {sw}v[0] : i32, {JV}")
        w(f"{ind}channel.publish {sw} : write<{J}>")

    def sig_wait(ch, ind):
        sr = w.t("sr")
        w(f"{ind}{sr}, {sr}v = channel.acquire {ch} : channel<{J}> -> (read<{J}>, {JV})")
        w(f"{ind}channel.release {sr} : read<{J}>")

    def chunk_begin(ind):
        st = {}
        if owner:
            ar = w.t("ar")
            w(f"{ind}{ar}, {ar}v = channel.acquire %ach{n} : channel<{I8}> -> (read<{I8}>, {V8})")
            st["a"] = ar
            sig_send(f"%job{pn}", ind)
        else:
            sig_wait(f"%job{pn}", ind)
        br = w.t("br")
        w(f"{ind}{br}, {br}v = channel.acquire %bch{n} : channel<{I8}> -> (read<{I8}>, {V8})")
        st["b"] = br
        return st

    def chunk_end(st, ind):
        w(f"{ind}channel.release {st['b']} : read<{I8}>")
        if owner:
            sig_wait(f"%ack{pn}", ind)
            w(f"{ind}channel.release {st['a']} : read<{I8}>")
        else:
            sig_send(f"%ack{pn}", ind)

    def drain(gloc, ind):
        # staging order: owner lo, peer lo, owner hi, peer hi -> y block [owner lo, owner hi, peer lo, peer hi]
        jj, rb, t, yb0 = w.t("y"), w.t("y"), w.t("y"), w.t("y")
        w(f"{ind}{jj} = index.add %j0, {gloc} : index")
        w(f"{ind}{rb} = index.add %prow, %rbl : index")
        w(f"{ind}{t} = index.madd {rb}, %tsteps, {jj} : index")
        w(f"{ind}{yb0} = index.scale {t}, %blkbytes : index, offset -> offset")
        rd, ye, cp, group = (w.t(v) for v in ("rd", "ye", "cp", "group"))
        w(f"{ind}{rd}, {rd}v = channel.acquire %ring{pn} : channel<{I32}> -> (read<{I32}>, {V32})")
        w(f"{ind}{ye} = buffer.view %y[{yb0}] : buffer -> {V32}")
        w(f"{ind}{cp} = kernel.async.copy {rd}v to {ye} {DMA_OUT} : {V32} to {V32} -> kernel.async.token")
        w(f"{ind}{group} = kernel.async.group {cp} : kernel.async.token -> kernel.async.group")
        w(f"{ind}kernel.async.wait {group} {{newer_groups = 0}} : kernel.async.group")
        w(f"{ind}channel.release {rd} : read<{I32}>")

    def tile(ind, gloc):
        clo, chi = w.t("clo"), w.t("chi")
        w(f"{ind}{clo} = index.constant {0 if owner else 2 * REC} : offset")
        w(f"{ind}{chi} = index.constant {REC if owner else 3 * REC} : offset")
        st = chunk_begin(ind)
        s0 = w.t("so")
        w(f"{ind}{s0} = index.constant 0 : offset")
        rw = w.t("rw")
        if owner:
            sig_wait(f"%cgo{pn}", ind)
        else:
            w(f"{ind}{rw}, {rw}v = channel.reserve %ring{n} : channel<{I32}> -> (write<{I32}>, {V32})")
            sig_send(f"%cgo{pn}", ind)
        # Preserve the SSA names in the qualified programs after folding the initial zero stores.
        w.n += 6
        calls(s0, clo, chi, (0, 1, 2, 3), ind, first=True)
        chunk_end(st, ind)
        k = w.t("k")
        w(f"{ind}scf.for {k} = [%c1 to %kcm1 step %c1] {{")
        i2 = ind + "  "
        st = chunk_begin(i2)
        par, so, sob = w.t("par"), w.t("so"), w.t("sob")
        w(f"{i2}{par} = index.andi {k}, %c1 : index")
        w(f"{i2}{so} = index.scale {par}, %astride : index, offset -> offset")
        w(f"{i2}{sob} = index.scale {par}, %bstride : index, offset -> offset")
        calls(so, clo, chi, (0, 1, 2, 3), i2, sob)
        chunk_end(st, i2)
        w(f"{ind}}}")
        # The final chunk is in slot 1 for both supported K dimensions.
        st = chunk_begin(ind)
        sl, slb = w.t("so"), w.t("sob")
        w(f"{ind}{sl} = index.add %astride, %origin : offset")
        w(f"{ind}{slb} = index.add %bstride, %origin : offset")
        calls(sl, clo, chi, (0, 1, 2, 3), ind, slb)
        if owner:
            sig_send(f"%odone{pn}", ind)
        else:
            sig_wait(f"%odone{pn}", ind)
            w(f"{ind}channel.publish {rw} : write<{I32}>")
        chunk_end(st, ind)

        if not owner:
            drain(gloc, ind)
    w("    scf.for %gl = [%c0 to %tg step %c1] {")
    tile("      ", "%gl")
    w("    }")
    w("  }")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("k", type=int, choices=(2304, 3072))
    parser.add_argument("groups", type=int, choices=(1, 2))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(gen(args.k, args.groups) + "\n")
