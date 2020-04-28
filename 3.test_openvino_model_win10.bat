:: set environment
call "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\bin\setupvars.bat"

for %%M in (efficientnet-b0, efficientnet-b1, efficientnet-b3, efficientnet-b4, efficientnet-b5, efficientnet-b6, efficientnet-b7) ^
do ( python test_openvino.py -m "model/%%M/%%M.xml" -i "./image/cat.1.jpg" -d "MYRIAD" --labels "./label/imagenet.labels" -nt 5)

call cmd.exe