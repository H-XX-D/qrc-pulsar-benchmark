# ==============================================================================
# Vitis / Vivado HLS Synthesis Script for Jacobi Symplectic Core
# Target: AMD / Xilinx Kintex UltraScale Space-Grade FPGA (XQRKU060)
# ==============================================================================

open_project -reset prj_jacobi_symplectic
set_top jacobi_symplectic_step

add_files jacobi_symplectic_core.cpp
add_files jacobi_symplectic_core.h

open_solution -reset "solution1"
set_part {xqrku060-cna1509-1M-m}
create_clock -period 6.667 -name default ;# 150 MHz target clock

# Synthesize the core
csynth_design

# Export IP for Vivado block design integration
# export_design -format ip_catalog -description "Jacobi 4th-Order Symplectic Core for Autonomous Orbit Determination"

exit
