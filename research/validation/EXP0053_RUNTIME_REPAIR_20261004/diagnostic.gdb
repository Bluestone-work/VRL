set pagination off
set debuginfod enabled off
run
info program
thread 1
bt 8
info registers
x/16i $pc-24
disassemble /r array_subtract
