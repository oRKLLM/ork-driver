/* silu_tmpl_diff — dump the int8 vs int16 SiLU template INPUT-surface regs (0x50xx) to find why the int16
 * SiLU reads EWCUBEH cube while int8 reads linear. No NPU. */
#include "ork_npu.h"
int main(void){ ork_dump_silu_templates(); return 0; }
