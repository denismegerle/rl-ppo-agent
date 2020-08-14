@echo off
call :treeProcess
goto :eof

:treeProcess
rem Do whatever you want here over the files of this subdir, for example:
for %%f in (*.v2) do 7z a %%f.7z %%f -mx9

for /D %%d in (*) do (
    cd %%d
    call :treeProcess
    cd ..
)
exit /b