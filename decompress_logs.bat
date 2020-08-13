@echo off
call :treeProcess
goto :eof

:treeProcess
rem Do whatever you want here over the files of this subdir, for example:
for %%f in (*.v2.7z) do 7z e %%f -sdel -aos

for /D %%d in (*) do (
    cd %%d
    call :treeProcess
    cd ..
)
exit /b