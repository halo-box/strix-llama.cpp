low.func.def noinline target<amd.xdna.aie2p.core> @mk(%abase: reg<aie2p.ep>, %oa0: reg<aie2p.er>, %oa1: reg<aie2p.er>, %bbase: reg<aie2p.ep>, %ob: reg<aie2p.er>, %cbase: reg<aie2p.ep>, %oc0: reg<aie2p.er>, %oc1: reg<aie2p.er>) asm {
  %cbase2 = copy %cbase : reg<aie2p.ep> -> reg<aie2p.ep>
  %m_pa = mov.modifier %oa0
  %pa = paddb.modifier %abase, %m_pa
  %m_pb = mov.modifier %ob
  %pb = paddb.modifier %bbase, %m_pb
  %m_pc0 = mov.modifier %oc0
  %pc0 = paddb.modifier %cbase, %m_pc0
  %m_pc1 = mov.modifier %oc1
  %pc1 = paddb.modifier %cbase2, %m_pc1
  %step = mov.i32 1024
  %mfwd = mov.modifier %step
  %twostep = mov.i32 2048
  %m2 = mov.modifier %twostep
  %cstep = mov.i32 512
  %mc = mov.modifier %cstep
  %nstart = mova.i32 4
  %mode = mova.i32 776
  low.br ^tile(%nstart: reg<aie2p.er>, %pb: reg<aie2p.ep>, %pc0: reg<aie2p.ep>, %pc1: reg<aie2p.ep>)
^tile(%count: reg<aie2p.er>, %bcur: reg<aie2p.ep>, %cc0: reg<aie2p.ep>, %cc1: reg<aie2p.ep>):
  %abase0 = copy %pa : reg<aie2p.ep> -> reg<aie2p.ep>
  %acopy = copy %pa : reg<aie2p.ep> -> reg<aie2p.ep>
  %abase1 = paddb.modifier %acopy, %mfwd
  %bcopy = copy %bcur : reg<aie2p.ep> -> reg<aie2p.ep>
  %pbc_1 = copy %bcopy : reg<aie2p.ep> -> reg<aie2p.ep>
  %pb1_2 = paddb.modifier %pbc_1, %mfwd
  %c_3 = vlda.acc %cc0, 0
  %c_4 = vlda.acc %cc0, 64
  %c_5 = vlda.acc %cc0, 128
  %c_6 = vlda.acc %cc0, 192
  %cfull_7 = concat(%c_3, %c_4, %c_5, %c_6) : (reg<aie2p.mbms>, reg<aie2p.mbms>, reg<aie2p.mbms>, reg<aie2p.mbms>) -> reg<aie2p.mbms x4>
  %c_8 = vlda.acc %cc0, 256
  %c_9 = vlda.acc %cc0, 320
  %c_10 = vlda.acc %cc0, 384
  %c_11 = vlda.acc %cc0, 448
  %cfull_12 = concat(%c_8, %c_9, %c_10, %c_11) : (reg<aie2p.mbms>, reg<aie2p.mbms>, reg<aie2p.mbms>, reg<aie2p.mbms>) -> reg<aie2p.mbms x4>
  %c_13 = vlda.acc %cc1, 0
  %c_14 = vlda.acc %cc1, 64
  %c_15 = vlda.acc %cc1, 128
  %c_16 = vlda.acc %cc1, 192
  %cfull_17 = concat(%c_13, %c_14, %c_15, %c_16) : (reg<aie2p.mbms>, reg<aie2p.mbms>, reg<aie2p.mbms>, reg<aie2p.mbms>) -> reg<aie2p.mbms x4>
  %c_18 = vlda.acc %cc1, 256
  %c_19 = vlda.acc %cc1, 320
  %c_20 = vlda.acc %cc1, 384
  %c_21 = vlda.acc %cc1, 448
  %cfull_22 = concat(%c_18, %c_19, %c_20, %c_21) : (reg<aie2p.mbms>, reg<aie2p.mbms>, reg<aie2p.mbms>, reg<aie2p.mbms>) -> reg<aie2p.mbms x4>
  %a0_0_0_23, %abase0_24 = vldb.512.i8x64.post %abase0, 64
  %b0_0_0_25, %pb0_26 = vldb.512.i8x64.post %bcopy, 64
  %a1_0_0_27, %abase1_28 = vldb.512.i8x64.post %abase1, 64
  %b1_0_0_29, %pb1_30 = vldb.512.i8x64.post %pb1_2, 64
  %a0_0_1_31, %abase0_32 = vldb.512.i8x64.post %abase0_24, 64
  %b0_0_1_33, %pb0_34 = vldb.512.i8x64.post %pb0_26, 64
  %a1_0_1_35, %abase1_36 = vldb.512.i8x64.post %abase1_28, 64
  %b1_0_1_37, %pb1_38 = vldb.512.i8x64.post %pb1_30, 64
  %acc00_39 = mma.s8s8.m8n8k8 %cfull_7, %a0_0_0_23, %b0_0_0_25, %mode
  %acc01_40 = mma.s8s8.m8n8k8 %cfull_12, %a0_0_0_23, %b1_0_0_29, %mode
  %a0_0_2_43, %abase0_44 = vldb.512.i8x64.post %abase0_32, 64
  %acc10_41 = mma.s8s8.m8n8k8 %cfull_17, %a1_0_0_27, %b0_0_0_25, %mode
  %b0_0_2_45, %pb0_46 = vldb.512.i8x64.post %pb0_34, 64
  %acc11_42 = mma.s8s8.m8n8k8 %cfull_22, %a1_0_0_27, %b1_0_0_29, %mode
  %a1_0_2_47, %abase1_48 = vldb.512.i8x64.post %abase1_36, 64
  %b1_0_2_49, %pb1_50 = vldb.512.i8x64.post %pb1_38, 64
  low.schedule.fence
  %acc00_51 = mma.s8s8.m8n8k8 %acc00_39, %a0_0_1_31, %b0_0_1_33, %mode
  %acc01_52 = mma.s8s8.m8n8k8 %acc01_40, %a0_0_1_31, %b1_0_1_37, %mode
  %a0_0_3_55, %abase0_56 = vldb.512.i8x64.post %abase0_44, 64
  %acc10_53 = mma.s8s8.m8n8k8 %acc10_41, %a1_0_1_35, %b0_0_1_33, %mode
  %b0_0_3_57, %pb0_58 = vldb.512.i8x64.post %pb0_46, 64
  %acc11_54 = mma.s8s8.m8n8k8 %acc11_42, %a1_0_1_35, %b1_0_1_37, %mode
  %a1_0_3_59, %abase1_60 = vldb.512.i8x64.post %abase1_48, 64
  %b1_0_3_61, %pb1_62 = vldb.512.i8x64.post %pb1_50, 64
  low.schedule.fence
  %acc00_63 = mma.s8s8.m8n8k8 %acc00_51, %a0_0_2_43, %b0_0_2_45, %mode
  %acc01_64 = mma.s8s8.m8n8k8 %acc01_52, %a0_0_2_43, %b1_0_2_49, %mode
  %a0_0_4_67, %abase0_68 = vlda.512.i8x64.post %abase0_56, 64
  %acc10_65 = mma.s8s8.m8n8k8 %acc10_53, %a1_0_2_47, %b0_0_2_45, %mode
  %b0_0_4_69, %pb0_70 = vldb.512.i8x64.post %pb0_58, 64
  %acc11_66 = mma.s8s8.m8n8k8 %acc11_54, %a1_0_2_47, %b1_0_2_49, %mode
  %a1_0_4_71, %abase1_72 = vlda.512.i8x64.post %abase1_60, 64
  %b1_0_4_73, %pb1_74 = vldb.512.i8x64.post %pb1_62, 64
  low.schedule.fence
  %acc00_75 = mma.s8s8.m8n8k8 %acc00_63, %a0_0_3_55, %b0_0_3_57, %mode
  %acc01_76 = mma.s8s8.m8n8k8 %acc01_64, %a0_0_3_55, %b1_0_3_61, %mode
  %a0_0_5_79, %abase0_80 = vlda.512.i8x64.post %abase0_68, 64
  %acc10_77 = mma.s8s8.m8n8k8 %acc10_65, %a1_0_3_59, %b0_0_3_57, %mode
  %b0_0_5_81, %pb0_82 = vldb.512.i8x64.post %pb0_70, 64
  %acc11_78 = mma.s8s8.m8n8k8 %acc11_66, %a1_0_3_59, %b1_0_3_61, %mode
  %a1_0_5_83, %abase1_84 = vlda.512.i8x64.post %abase1_72, 64
  %b1_0_5_85, %pb1_86 = vldb.512.i8x64.post %pb1_74, 64
  low.schedule.fence
  %acc00_87 = mma.s8s8.m8n8k8 %acc00_75, %a0_0_4_67, %b0_0_4_69, %mode
  %acc01_88 = mma.s8s8.m8n8k8 %acc01_76, %a0_0_4_67, %b1_0_4_73, %mode
  %a0_0_6_91, %abase0_92 = vlda.512.i8x64.post %abase0_80, 64
  %acc10_89 = mma.s8s8.m8n8k8 %acc10_77, %a1_0_4_71, %b0_0_4_69, %mode
  %b0_0_6_93, %pb0_94 = vldb.512.i8x64.post %pb0_82, 64
  %acc11_90 = mma.s8s8.m8n8k8 %acc11_78, %a1_0_4_71, %b1_0_4_73, %mode
  %a1_0_6_95, %abase1_96 = vlda.512.i8x64.post %abase1_84, 64
  %b1_0_6_97, %pb1_98 = vldb.512.i8x64.post %pb1_86, 64
  low.schedule.fence
  %acc00_99 = mma.s8s8.m8n8k8 %acc00_87, %a0_0_5_79, %b0_0_5_81, %mode
  %acc01_100 = mma.s8s8.m8n8k8 %acc01_88, %a0_0_5_79, %b1_0_5_85, %mode
  %a0_0_7_103, %abase0_104 = vlda.512.i8x64.post %abase0_92, 64
  %acc10_101 = mma.s8s8.m8n8k8 %acc10_89, %a1_0_5_83, %b0_0_5_81, %mode
  %b0_0_7_105, %pb0_106 = vldb.512.i8x64.post %pb0_94, 64
  %acc11_102 = mma.s8s8.m8n8k8 %acc11_90, %a1_0_5_83, %b1_0_5_85, %mode
  %a1_0_7_107, %abase1_108 = vlda.512.i8x64.post %abase1_96, 64
  %b1_0_7_109, %pb1_110 = vldb.512.i8x64.post %pb1_98, 64
  low.schedule.fence
  %acc00_111 = mma.s8s8.m8n8k8 %acc00_99, %a0_0_6_91, %b0_0_6_93, %mode
  %acc01_112 = mma.s8s8.m8n8k8 %acc01_100, %a0_0_6_91, %b1_0_6_97, %mode
  %a0_0_8_115, %abase0_116 = vlda.512.i8x64.post %abase0_104, 64
  %acc10_113 = mma.s8s8.m8n8k8 %acc10_101, %a1_0_6_95, %b0_0_6_93, %mode
  %b0_0_8_117, %pb0_118 = vldb.512.i8x64.post %pb0_106, 64
  %acc11_114 = mma.s8s8.m8n8k8 %acc11_102, %a1_0_6_95, %b1_0_6_97, %mode
  %a1_0_8_119, %abase1_120 = vlda.512.i8x64.post %abase1_108, 64
  %b1_0_8_121, %pb1_122 = vldb.512.i8x64.post %pb1_110, 64
  low.schedule.fence
  %acc00_123 = mma.s8s8.m8n8k8 %acc00_111, %a0_0_7_103, %b0_0_7_105, %mode
  %acc01_124 = mma.s8s8.m8n8k8 %acc01_112, %a0_0_7_103, %b1_0_7_109, %mode
  %a0_0_9_127, %abase0_128 = vlda.512.i8x64.post %abase0_116, 64
  %acc10_125 = mma.s8s8.m8n8k8 %acc10_113, %a1_0_7_107, %b0_0_7_105, %mode
  %b0_0_9_129, %pb0_130 = vldb.512.i8x64.post %pb0_118, 64
  %acc11_126 = mma.s8s8.m8n8k8 %acc11_114, %a1_0_7_107, %b1_0_7_109, %mode
  %a1_0_9_131, %abase1_132 = vlda.512.i8x64.post %abase1_120, 64
  %b1_0_9_133, %pb1_134 = vldb.512.i8x64.post %pb1_122, 64
  low.schedule.fence
  %acc00_135 = mma.s8s8.m8n8k8 %acc00_123, %a0_0_8_115, %b0_0_8_117, %mode
  %acc01_136 = mma.s8s8.m8n8k8 %acc01_124, %a0_0_8_115, %b1_0_8_121, %mode
  %a0_0_10_139, %abase0_140 = vlda.512.i8x64.post %abase0_128, 64
  %acc10_137 = mma.s8s8.m8n8k8 %acc10_125, %a1_0_8_119, %b0_0_8_117, %mode
  %b0_0_10_141, %pb0_142 = vldb.512.i8x64.post %pb0_130, 64
  %acc11_138 = mma.s8s8.m8n8k8 %acc11_126, %a1_0_8_119, %b1_0_8_121, %mode
  %a1_0_10_143, %abase1_144 = vlda.512.i8x64.post %abase1_132, 64
  %b1_0_10_145, %pb1_146 = vldb.512.i8x64.post %pb1_134, 64
  low.schedule.fence
  %acc00_147 = mma.s8s8.m8n8k8 %acc00_135, %a0_0_9_127, %b0_0_9_129, %mode
  %acc01_148 = mma.s8s8.m8n8k8 %acc01_136, %a0_0_9_127, %b1_0_9_133, %mode
  %a0_0_11_151, %abase0_152 = vlda.512.i8x64.post %abase0_140, 64
  %acc10_149 = mma.s8s8.m8n8k8 %acc10_137, %a1_0_9_131, %b0_0_9_129, %mode
  %b0_0_11_153, %pb0_154 = vldb.512.i8x64.post %pb0_142, 64
  %acc11_150 = mma.s8s8.m8n8k8 %acc11_138, %a1_0_9_131, %b1_0_9_133, %mode
  %a1_0_11_155, %abase1_156 = vlda.512.i8x64.post %abase1_144, 64
  %b1_0_11_157, %pb1_158 = vldb.512.i8x64.post %pb1_146, 64
  low.schedule.fence
  %acc00_159 = mma.s8s8.m8n8k8 %acc00_147, %a0_0_10_139, %b0_0_10_141, %mode
  %acc01_160 = mma.s8s8.m8n8k8 %acc01_148, %a0_0_10_139, %b1_0_10_145, %mode
  %a0_0_12_163, %abase0_164 = vlda.512.i8x64.post %abase0_152, 64
  %acc10_161 = mma.s8s8.m8n8k8 %acc10_149, %a1_0_10_143, %b0_0_10_141, %mode
  %b0_0_12_165, %pb0_166 = vldb.512.i8x64.post %pb0_154, 64
  %acc11_162 = mma.s8s8.m8n8k8 %acc11_150, %a1_0_10_143, %b1_0_10_145, %mode
  %a1_0_12_167, %abase1_168 = vlda.512.i8x64.post %abase1_156, 64
  %b1_0_12_169, %pb1_170 = vldb.512.i8x64.post %pb1_158, 64
  low.schedule.fence
  %acc00_171 = mma.s8s8.m8n8k8 %acc00_159, %a0_0_11_151, %b0_0_11_153, %mode
  %acc01_172 = mma.s8s8.m8n8k8 %acc01_160, %a0_0_11_151, %b1_0_11_157, %mode
  %a0_0_13_175, %abase0_176 = vlda.512.i8x64.post %abase0_164, 64
  %acc10_173 = mma.s8s8.m8n8k8 %acc10_161, %a1_0_11_155, %b0_0_11_153, %mode
  %b0_0_13_177, %pb0_178 = vldb.512.i8x64.post %pb0_166, 64
  %acc11_174 = mma.s8s8.m8n8k8 %acc11_162, %a1_0_11_155, %b1_0_11_157, %mode
  %a1_0_13_179, %abase1_180 = vlda.512.i8x64.post %abase1_168, 64
  %b1_0_13_181, %pb1_182 = vldb.512.i8x64.post %pb1_170, 64
  low.schedule.fence
  %acc00_183 = mma.s8s8.m8n8k8 %acc00_171, %a0_0_12_163, %b0_0_12_165, %mode
  %acc01_184 = mma.s8s8.m8n8k8 %acc01_172, %a0_0_12_163, %b1_0_12_169, %mode
  %a0_0_14_187, %abase0_188 = vlda.512.i8x64.post %abase0_176, 64
  %acc10_185 = mma.s8s8.m8n8k8 %acc10_173, %a1_0_12_167, %b0_0_12_165, %mode
  %b0_0_14_189, %pb0_190 = vldb.512.i8x64.post %pb0_178, 64
  %acc11_186 = mma.s8s8.m8n8k8 %acc11_174, %a1_0_12_167, %b1_0_12_169, %mode
  %a1_0_14_191, %abase1_192 = vlda.512.i8x64.post %abase1_180, 64
  %b1_0_14_193, %pb1_194 = vldb.512.i8x64.post %pb1_182, 64
  low.schedule.fence
  %acc00_195 = mma.s8s8.m8n8k8 %acc00_183, %a0_0_13_175, %b0_0_13_177, %mode
  %acc01_196 = mma.s8s8.m8n8k8 %acc01_184, %a0_0_13_175, %b1_0_13_181, %mode
  %a0_0_15_199, %abase0_200 = vlda.512.i8x64.post %abase0_188, 64
  %acc10_197 = mma.s8s8.m8n8k8 %acc10_185, %a1_0_13_179, %b0_0_13_177, %mode
  %b0_0_15_201, %pb0_202 = vldb.512.i8x64.post %pb0_190, 64
  %acc11_198 = mma.s8s8.m8n8k8 %acc11_186, %a1_0_13_179, %b1_0_13_181, %mode
  %a1_0_15_203, %abase1_204 = vlda.512.i8x64.post %abase1_192, 64
  %b1_0_15_205, %pb1_206 = vldb.512.i8x64.post %pb1_194, 64
  low.schedule.fence
  %acc00_207 = mma.s8s8.m8n8k8 %acc00_195, %a0_0_14_187, %b0_0_14_189, %mode
  %acc01_208 = mma.s8s8.m8n8k8 %acc01_196, %a0_0_14_187, %b1_0_14_193, %mode
  %acc10_209 = mma.s8s8.m8n8k8 %acc10_197, %a1_0_14_191, %b0_0_14_189, %mode
  %acc11_210 = mma.s8s8.m8n8k8 %acc11_198, %a1_0_14_191, %b1_0_14_193, %mode
  low.schedule.fence
  %acc00_211 = mma.s8s8.m8n8k8 %acc00_207, %a0_0_15_199, %b0_0_15_201, %mode
  %acc01_212 = mma.s8s8.m8n8k8 %acc01_208, %a0_0_15_199, %b1_0_15_205, %mode
  %acc10_213 = mma.s8s8.m8n8k8 %acc10_209, %a1_0_15_203, %b0_0_15_201, %mode
  %acc11_214 = mma.s8s8.m8n8k8 %acc11_210, %a1_0_15_203, %b1_0_15_205, %mode
  %sl_215 = slice %acc00_211[0] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_215, %cc0, 0
  %sl_216 = slice %acc00_211[1] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_216, %cc0, 64
  %sl_217 = slice %acc00_211[2] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_217, %cc0, 128
  %sl_218 = slice %acc00_211[3] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_218, %cc0, 192
  %sl_219 = slice %acc01_212[0] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_219, %cc0, 256
  %sl_220 = slice %acc01_212[1] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_220, %cc0, 320
  %sl_221 = slice %acc01_212[2] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_221, %cc0, 384
  %sl_222 = slice %acc01_212[3] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_222, %cc0, 448
  %sl_223 = slice %acc10_213[0] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_223, %cc1, 0
  %sl_224 = slice %acc10_213[1] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_224, %cc1, 64
  %sl_225 = slice %acc10_213[2] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_225, %cc1, 128
  %sl_226 = slice %acc10_213[3] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_226, %cc1, 192
  %sl_227 = slice %acc11_214[0] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_227, %cc1, 256
  %sl_228 = slice %acc11_214[1] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_228, %cc1, 320
  %sl_229 = slice %acc11_214[2] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_229, %cc1, 384
  %sl_230 = slice %acc11_214[3] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_230, %cc1, 448
  %remaining = add %count, -1
  low.cond_br %remaining, ^next, ^done : reg<aie2p.er>
^next:
  %bn = paddb.modifier %bcur, %m2
  %cn0 = padda.modifier %cc0, %mc
  %cn1 = padda.modifier %cc1, %mc
  low.br ^tile(%remaining: reg<aie2p.er>, %bn: reg<aie2p.ep>, %cn0: reg<aie2p.ep>, %cn1: reg<aie2p.ep>)
^done:
  return
}
low.func.def noinline target<amd.xdna.aie2p.core> @mk0(%abase: reg<aie2p.ep>, %oa0: reg<aie2p.er>, %oa1: reg<aie2p.er>, %bbase: reg<aie2p.ep>, %ob: reg<aie2p.er>, %cbase: reg<aie2p.ep>, %oc0: reg<aie2p.er>, %oc1: reg<aie2p.er>) asm {
  %cbase2 = copy %cbase : reg<aie2p.ep> -> reg<aie2p.ep>
  %m_pa = mov.modifier %oa0
  %pa = paddb.modifier %abase, %m_pa
  %m_pb = mov.modifier %ob
  %pb = paddb.modifier %bbase, %m_pb
  %m_pc0 = mov.modifier %oc0
  %pc0 = paddb.modifier %cbase, %m_pc0
  %m_pc1 = mov.modifier %oc1
  %pc1 = paddb.modifier %cbase2, %m_pc1
  %step = mov.i32 1024
  %mfwd = mov.modifier %step
  %twostep = mov.i32 2048
  %m2 = mov.modifier %twostep
  %cstep = mov.i32 512
  %mc = mov.modifier %cstep
  %nstart = mova.i32 4
  %mode = mova.i32 776
  low.br ^tile(%nstart: reg<aie2p.er>, %pb: reg<aie2p.ep>, %pc0: reg<aie2p.ep>, %pc1: reg<aie2p.ep>)
^tile(%count: reg<aie2p.er>, %bcur: reg<aie2p.ep>, %cc0: reg<aie2p.ep>, %cc1: reg<aie2p.ep>):
  %abase0 = copy %pa : reg<aie2p.ep> -> reg<aie2p.ep>
  %acopy = copy %pa : reg<aie2p.ep> -> reg<aie2p.ep>
  %abase1 = paddb.modifier %acopy, %mfwd
  %bcopy = copy %bcur : reg<aie2p.ep> -> reg<aie2p.ep>
  %pbc_1 = copy %bcopy : reg<aie2p.ep> -> reg<aie2p.ep>
  %pb1_2 = paddb.modifier %pbc_1, %mfwd
  %a0_0_0_23, %abase0_24 = vldb.512.i8x64.post %abase0, 64
  %b0_0_0_25, %pb0_26 = vldb.512.i8x64.post %bcopy, 64
  %a1_0_0_27, %abase1_28 = vldb.512.i8x64.post %abase1, 64
  %b1_0_0_29, %pb1_30 = vldb.512.i8x64.post %pb1_2, 64
  %a0_0_1_31, %abase0_32 = vldb.512.i8x64.post %abase0_24, 64
  %b0_0_1_33, %pb0_34 = vldb.512.i8x64.post %pb0_26, 64
  %a1_0_1_35, %abase1_36 = vldb.512.i8x64.post %abase1_28, 64
  %b1_0_1_37, %pb1_38 = vldb.512.i8x64.post %pb1_30, 64
  %acc00_39 = mmul.s8s8.m8n8k8 %a0_0_0_23, %b0_0_0_25, %mode
  %acc01_40 = mmul.s8s8.m8n8k8 %a0_0_0_23, %b1_0_0_29, %mode
  %a0_0_2_43, %abase0_44 = vldb.512.i8x64.post %abase0_32, 64
  %acc10_41 = mmul.s8s8.m8n8k8 %a1_0_0_27, %b0_0_0_25, %mode
  %b0_0_2_45, %pb0_46 = vldb.512.i8x64.post %pb0_34, 64
  %acc11_42 = mmul.s8s8.m8n8k8 %a1_0_0_27, %b1_0_0_29, %mode
  %a1_0_2_47, %abase1_48 = vldb.512.i8x64.post %abase1_36, 64
  %b1_0_2_49, %pb1_50 = vldb.512.i8x64.post %pb1_38, 64
  low.schedule.fence
  %acc00_51 = mma.s8s8.m8n8k8 %acc00_39, %a0_0_1_31, %b0_0_1_33, %mode
  %acc01_52 = mma.s8s8.m8n8k8 %acc01_40, %a0_0_1_31, %b1_0_1_37, %mode
  %a0_0_3_55, %abase0_56 = vldb.512.i8x64.post %abase0_44, 64
  %acc10_53 = mma.s8s8.m8n8k8 %acc10_41, %a1_0_1_35, %b0_0_1_33, %mode
  %b0_0_3_57, %pb0_58 = vldb.512.i8x64.post %pb0_46, 64
  %acc11_54 = mma.s8s8.m8n8k8 %acc11_42, %a1_0_1_35, %b1_0_1_37, %mode
  %a1_0_3_59, %abase1_60 = vldb.512.i8x64.post %abase1_48, 64
  %b1_0_3_61, %pb1_62 = vldb.512.i8x64.post %pb1_50, 64
  low.schedule.fence
  %acc00_63 = mma.s8s8.m8n8k8 %acc00_51, %a0_0_2_43, %b0_0_2_45, %mode
  %acc01_64 = mma.s8s8.m8n8k8 %acc01_52, %a0_0_2_43, %b1_0_2_49, %mode
  %a0_0_4_67, %abase0_68 = vlda.512.i8x64.post %abase0_56, 64
  %acc10_65 = mma.s8s8.m8n8k8 %acc10_53, %a1_0_2_47, %b0_0_2_45, %mode
  %b0_0_4_69, %pb0_70 = vldb.512.i8x64.post %pb0_58, 64
  %acc11_66 = mma.s8s8.m8n8k8 %acc11_54, %a1_0_2_47, %b1_0_2_49, %mode
  %a1_0_4_71, %abase1_72 = vlda.512.i8x64.post %abase1_60, 64
  %b1_0_4_73, %pb1_74 = vldb.512.i8x64.post %pb1_62, 64
  low.schedule.fence
  %acc00_75 = mma.s8s8.m8n8k8 %acc00_63, %a0_0_3_55, %b0_0_3_57, %mode
  %acc01_76 = mma.s8s8.m8n8k8 %acc01_64, %a0_0_3_55, %b1_0_3_61, %mode
  %a0_0_5_79, %abase0_80 = vlda.512.i8x64.post %abase0_68, 64
  %acc10_77 = mma.s8s8.m8n8k8 %acc10_65, %a1_0_3_59, %b0_0_3_57, %mode
  %b0_0_5_81, %pb0_82 = vldb.512.i8x64.post %pb0_70, 64
  %acc11_78 = mma.s8s8.m8n8k8 %acc11_66, %a1_0_3_59, %b1_0_3_61, %mode
  %a1_0_5_83, %abase1_84 = vlda.512.i8x64.post %abase1_72, 64
  %b1_0_5_85, %pb1_86 = vldb.512.i8x64.post %pb1_74, 64
  low.schedule.fence
  %acc00_87 = mma.s8s8.m8n8k8 %acc00_75, %a0_0_4_67, %b0_0_4_69, %mode
  %acc01_88 = mma.s8s8.m8n8k8 %acc01_76, %a0_0_4_67, %b1_0_4_73, %mode
  %a0_0_6_91, %abase0_92 = vlda.512.i8x64.post %abase0_80, 64
  %acc10_89 = mma.s8s8.m8n8k8 %acc10_77, %a1_0_4_71, %b0_0_4_69, %mode
  %b0_0_6_93, %pb0_94 = vldb.512.i8x64.post %pb0_82, 64
  %acc11_90 = mma.s8s8.m8n8k8 %acc11_78, %a1_0_4_71, %b1_0_4_73, %mode
  %a1_0_6_95, %abase1_96 = vlda.512.i8x64.post %abase1_84, 64
  %b1_0_6_97, %pb1_98 = vldb.512.i8x64.post %pb1_86, 64
  low.schedule.fence
  %acc00_99 = mma.s8s8.m8n8k8 %acc00_87, %a0_0_5_79, %b0_0_5_81, %mode
  %acc01_100 = mma.s8s8.m8n8k8 %acc01_88, %a0_0_5_79, %b1_0_5_85, %mode
  %a0_0_7_103, %abase0_104 = vlda.512.i8x64.post %abase0_92, 64
  %acc10_101 = mma.s8s8.m8n8k8 %acc10_89, %a1_0_5_83, %b0_0_5_81, %mode
  %b0_0_7_105, %pb0_106 = vldb.512.i8x64.post %pb0_94, 64
  %acc11_102 = mma.s8s8.m8n8k8 %acc11_90, %a1_0_5_83, %b1_0_5_85, %mode
  %a1_0_7_107, %abase1_108 = vlda.512.i8x64.post %abase1_96, 64
  %b1_0_7_109, %pb1_110 = vldb.512.i8x64.post %pb1_98, 64
  low.schedule.fence
  %acc00_111 = mma.s8s8.m8n8k8 %acc00_99, %a0_0_6_91, %b0_0_6_93, %mode
  %acc01_112 = mma.s8s8.m8n8k8 %acc01_100, %a0_0_6_91, %b1_0_6_97, %mode
  %a0_0_8_115, %abase0_116 = vlda.512.i8x64.post %abase0_104, 64
  %acc10_113 = mma.s8s8.m8n8k8 %acc10_101, %a1_0_6_95, %b0_0_6_93, %mode
  %b0_0_8_117, %pb0_118 = vldb.512.i8x64.post %pb0_106, 64
  %acc11_114 = mma.s8s8.m8n8k8 %acc11_102, %a1_0_6_95, %b1_0_6_97, %mode
  %a1_0_8_119, %abase1_120 = vlda.512.i8x64.post %abase1_108, 64
  %b1_0_8_121, %pb1_122 = vldb.512.i8x64.post %pb1_110, 64
  low.schedule.fence
  %acc00_123 = mma.s8s8.m8n8k8 %acc00_111, %a0_0_7_103, %b0_0_7_105, %mode
  %acc01_124 = mma.s8s8.m8n8k8 %acc01_112, %a0_0_7_103, %b1_0_7_109, %mode
  %a0_0_9_127, %abase0_128 = vlda.512.i8x64.post %abase0_116, 64
  %acc10_125 = mma.s8s8.m8n8k8 %acc10_113, %a1_0_7_107, %b0_0_7_105, %mode
  %b0_0_9_129, %pb0_130 = vldb.512.i8x64.post %pb0_118, 64
  %acc11_126 = mma.s8s8.m8n8k8 %acc11_114, %a1_0_7_107, %b1_0_7_109, %mode
  %a1_0_9_131, %abase1_132 = vlda.512.i8x64.post %abase1_120, 64
  %b1_0_9_133, %pb1_134 = vldb.512.i8x64.post %pb1_122, 64
  low.schedule.fence
  %acc00_135 = mma.s8s8.m8n8k8 %acc00_123, %a0_0_8_115, %b0_0_8_117, %mode
  %acc01_136 = mma.s8s8.m8n8k8 %acc01_124, %a0_0_8_115, %b1_0_8_121, %mode
  %a0_0_10_139, %abase0_140 = vlda.512.i8x64.post %abase0_128, 64
  %acc10_137 = mma.s8s8.m8n8k8 %acc10_125, %a1_0_8_119, %b0_0_8_117, %mode
  %b0_0_10_141, %pb0_142 = vldb.512.i8x64.post %pb0_130, 64
  %acc11_138 = mma.s8s8.m8n8k8 %acc11_126, %a1_0_8_119, %b1_0_8_121, %mode
  %a1_0_10_143, %abase1_144 = vlda.512.i8x64.post %abase1_132, 64
  %b1_0_10_145, %pb1_146 = vldb.512.i8x64.post %pb1_134, 64
  low.schedule.fence
  %acc00_147 = mma.s8s8.m8n8k8 %acc00_135, %a0_0_9_127, %b0_0_9_129, %mode
  %acc01_148 = mma.s8s8.m8n8k8 %acc01_136, %a0_0_9_127, %b1_0_9_133, %mode
  %a0_0_11_151, %abase0_152 = vlda.512.i8x64.post %abase0_140, 64
  %acc10_149 = mma.s8s8.m8n8k8 %acc10_137, %a1_0_9_131, %b0_0_9_129, %mode
  %b0_0_11_153, %pb0_154 = vldb.512.i8x64.post %pb0_142, 64
  %acc11_150 = mma.s8s8.m8n8k8 %acc11_138, %a1_0_9_131, %b1_0_9_133, %mode
  %a1_0_11_155, %abase1_156 = vlda.512.i8x64.post %abase1_144, 64
  %b1_0_11_157, %pb1_158 = vldb.512.i8x64.post %pb1_146, 64
  low.schedule.fence
  %acc00_159 = mma.s8s8.m8n8k8 %acc00_147, %a0_0_10_139, %b0_0_10_141, %mode
  %acc01_160 = mma.s8s8.m8n8k8 %acc01_148, %a0_0_10_139, %b1_0_10_145, %mode
  %a0_0_12_163, %abase0_164 = vlda.512.i8x64.post %abase0_152, 64
  %acc10_161 = mma.s8s8.m8n8k8 %acc10_149, %a1_0_10_143, %b0_0_10_141, %mode
  %b0_0_12_165, %pb0_166 = vldb.512.i8x64.post %pb0_154, 64
  %acc11_162 = mma.s8s8.m8n8k8 %acc11_150, %a1_0_10_143, %b1_0_10_145, %mode
  %a1_0_12_167, %abase1_168 = vlda.512.i8x64.post %abase1_156, 64
  %b1_0_12_169, %pb1_170 = vldb.512.i8x64.post %pb1_158, 64
  low.schedule.fence
  %acc00_171 = mma.s8s8.m8n8k8 %acc00_159, %a0_0_11_151, %b0_0_11_153, %mode
  %acc01_172 = mma.s8s8.m8n8k8 %acc01_160, %a0_0_11_151, %b1_0_11_157, %mode
  %a0_0_13_175, %abase0_176 = vlda.512.i8x64.post %abase0_164, 64
  %acc10_173 = mma.s8s8.m8n8k8 %acc10_161, %a1_0_11_155, %b0_0_11_153, %mode
  %b0_0_13_177, %pb0_178 = vldb.512.i8x64.post %pb0_166, 64
  %acc11_174 = mma.s8s8.m8n8k8 %acc11_162, %a1_0_11_155, %b1_0_11_157, %mode
  %a1_0_13_179, %abase1_180 = vlda.512.i8x64.post %abase1_168, 64
  %b1_0_13_181, %pb1_182 = vldb.512.i8x64.post %pb1_170, 64
  low.schedule.fence
  %acc00_183 = mma.s8s8.m8n8k8 %acc00_171, %a0_0_12_163, %b0_0_12_165, %mode
  %acc01_184 = mma.s8s8.m8n8k8 %acc01_172, %a0_0_12_163, %b1_0_12_169, %mode
  %a0_0_14_187, %abase0_188 = vlda.512.i8x64.post %abase0_176, 64
  %acc10_185 = mma.s8s8.m8n8k8 %acc10_173, %a1_0_12_167, %b0_0_12_165, %mode
  %b0_0_14_189, %pb0_190 = vldb.512.i8x64.post %pb0_178, 64
  %acc11_186 = mma.s8s8.m8n8k8 %acc11_174, %a1_0_12_167, %b1_0_12_169, %mode
  %a1_0_14_191, %abase1_192 = vlda.512.i8x64.post %abase1_180, 64
  %b1_0_14_193, %pb1_194 = vldb.512.i8x64.post %pb1_182, 64
  low.schedule.fence
  %acc00_195 = mma.s8s8.m8n8k8 %acc00_183, %a0_0_13_175, %b0_0_13_177, %mode
  %acc01_196 = mma.s8s8.m8n8k8 %acc01_184, %a0_0_13_175, %b1_0_13_181, %mode
  %a0_0_15_199, %abase0_200 = vlda.512.i8x64.post %abase0_188, 64
  %acc10_197 = mma.s8s8.m8n8k8 %acc10_185, %a1_0_13_179, %b0_0_13_177, %mode
  %b0_0_15_201, %pb0_202 = vldb.512.i8x64.post %pb0_190, 64
  %acc11_198 = mma.s8s8.m8n8k8 %acc11_186, %a1_0_13_179, %b1_0_13_181, %mode
  %a1_0_15_203, %abase1_204 = vlda.512.i8x64.post %abase1_192, 64
  %b1_0_15_205, %pb1_206 = vldb.512.i8x64.post %pb1_194, 64
  low.schedule.fence
  %acc00_207 = mma.s8s8.m8n8k8 %acc00_195, %a0_0_14_187, %b0_0_14_189, %mode
  %acc01_208 = mma.s8s8.m8n8k8 %acc01_196, %a0_0_14_187, %b1_0_14_193, %mode
  %acc10_209 = mma.s8s8.m8n8k8 %acc10_197, %a1_0_14_191, %b0_0_14_189, %mode
  %acc11_210 = mma.s8s8.m8n8k8 %acc11_198, %a1_0_14_191, %b1_0_14_193, %mode
  low.schedule.fence
  %acc00_211 = mma.s8s8.m8n8k8 %acc00_207, %a0_0_15_199, %b0_0_15_201, %mode
  %acc01_212 = mma.s8s8.m8n8k8 %acc01_208, %a0_0_15_199, %b1_0_15_205, %mode
  %acc10_213 = mma.s8s8.m8n8k8 %acc10_209, %a1_0_15_203, %b0_0_15_201, %mode
  %acc11_214 = mma.s8s8.m8n8k8 %acc11_210, %a1_0_15_203, %b1_0_15_205, %mode
  %sl_215 = slice %acc00_211[0] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_215, %cc0, 0
  %sl_216 = slice %acc00_211[1] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_216, %cc0, 64
  %sl_217 = slice %acc00_211[2] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_217, %cc0, 128
  %sl_218 = slice %acc00_211[3] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_218, %cc0, 192
  %sl_219 = slice %acc01_212[0] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_219, %cc0, 256
  %sl_220 = slice %acc01_212[1] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_220, %cc0, 320
  %sl_221 = slice %acc01_212[2] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_221, %cc0, 384
  %sl_222 = slice %acc01_212[3] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_222, %cc0, 448
  %sl_223 = slice %acc10_213[0] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_223, %cc1, 0
  %sl_224 = slice %acc10_213[1] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_224, %cc1, 64
  %sl_225 = slice %acc10_213[2] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_225, %cc1, 128
  %sl_226 = slice %acc10_213[3] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_226, %cc1, 192
  %sl_227 = slice %acc11_214[0] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_227, %cc1, 256
  %sl_228 = slice %acc11_214[1] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_228, %cc1, 320
  %sl_229 = slice %acc11_214[2] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_229, %cc1, 384
  %sl_230 = slice %acc11_214[3] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>
  vst.acc %sl_230, %cc1, 448
  %remaining = add %count, -1
  low.cond_br %remaining, ^next, ^done : reg<aie2p.er>
^next:
  %bn = paddb.modifier %bcur, %m2
  %cn0 = padda.modifier %cc0, %mc
  %cn1 = padda.modifier %cc1, %mc
  low.br ^tile(%remaining: reg<aie2p.er>, %bn: reg<aie2p.ep>, %cn0: reg<aie2p.ep>, %cn1: reg<aie2p.ep>)
^done:
  return
}
