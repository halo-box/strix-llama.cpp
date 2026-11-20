# embed a SPIR-V file as a C++ array: -DINPUT=x.spv -DOUTPUT=x.spv.h -DNAME=symbol
file(READ ${INPUT} hex HEX)
string(LENGTH "${hex}" len)
math(EXPR len "${len} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
file(WRITE ${OUTPUT} "#pragma once\n#include <cstddef>\nalignas(4) static const unsigned char ${NAME}_data[] = {${bytes}};\nstatic const size_t ${NAME}_len = ${len};\n")
