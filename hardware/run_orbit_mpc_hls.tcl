# Device remains user-selected. Clock is a request, not a measured result.
if {![info exists ::env(FPGA_PART)]} {error "Set FPGA_PART to the selected device"}
set here [file dirname [file normalize [info script]]]
set bits 32
if {[info exists ::env(QRC_MPC_BITS)]} {set bits $::env(QRC_MPC_BITS)}
set fraction [expr {$bits - 8}]
if {[info exists ::env(QRC_MPC_FRAC)]} {set fraction $::env(QRC_MPC_FRAC)}
if {![string is integer -strict $bits] || $bits < 16 || $bits > 32} {error "QRC_MPC_BITS must be an integer from 16 through 32"}
if {![string is integer -strict $fraction] || $fraction <= 0 || $fraction >= $bits - 1} {error "Invalid QRC_MPC_FRAC"}
set clock_ns 10.0
if {[info exists ::env(FPGA_CLOCK_NS)]} {set clock_ns $::env(FPGA_CLOCK_NS)}
set flags "-std=c++11 -DQRC_MPC_BITS=$bits -DQRC_MPC_FRAC=$fraction -DQRC_MPC_ITERATIONS=256"
open_project -reset orbit_mpc_hls
set_top orbit_mpc_solve
add_files $here/orbit_mpc_core.cpp -cflags $flags
add_files -tb $here/orbit_mpc_testbench.cpp -cflags $flags
open_solution -reset solution1
set_part $::env(FPGA_PART)
create_clock -period $clock_ns -name default
csim_design
csynth_design
cosim_design
exit
