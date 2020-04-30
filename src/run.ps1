$script="ida\extract_binary.py --prog_name lua --prog_ver 5.2.3 --cc gcc --cc_ver 5 --opt O0 --obf sub3"
$bfile="ida\.out\lua"

C:\\Users\\ligengwang\\IDA7.2\\idat64.exe -S"$script" "$bfile"
