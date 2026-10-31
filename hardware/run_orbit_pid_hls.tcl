# Device is intentionally selected by the caller, not guessed from a mission.
if {![info exists ::env(FPGA_PART)]} {error "Set FPGA_PART to the selected device"}
set here [file dirname [file normalize [info script]]]
set clock_ns 10.0
if {[info exists ::env(FPGA_CLOCK_NS)]} {set clock_ns $::env(FPGA_CLOCK_NS)}
open_project -reset orbit_pid_hls
set_top orbit_pid_step
add_files $here/orbit_pid_core.cpp -cflags "-std=c++11"
add_files -tb $here/orbit_pid_testbench.cpp -cflags "-std=c++11"
open_solution -reset solution1
set_part $::env(FPGA_PART)
create_clock -period $clock_ns -name default
csim_design
csynth_design
cosim_design
exit
