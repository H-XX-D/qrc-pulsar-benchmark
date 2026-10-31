# Optional synthesis recipe; host tests do not establish timing/resources.
open_project -reset cislunar_force_hls
set_top cislunar_verlet_step
add_files cislunar_force_core.cpp
open_solution -reset solution1
set_part {xqrku060-cna1509-1M-m}
create_clock -period 6.667 -name default
csynth_design
exit
