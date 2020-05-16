foreach ($file in dir "\\Mac\Home\Downloads\opensource\asm2vec_rebuild\bin\*-none")
{
    $script="src\ida\extract_binary.py --filepath=$file"
    echo C:\\Users\\ligengwang\\IDA7.2\\idat64.exe -B -S"$script" "$file"
    C:\\Users\\ligengwang\\IDA7.2\\idat64.exe -B -S"$script" "$file"
}
