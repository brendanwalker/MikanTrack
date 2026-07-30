@echo off
setlocal

IF NOT EXIST build mkdir build
pushd build
cmake .. -G "Visual Studio 17 2022" -A x64
IF %ERRORLEVEL% NEQ 0 (
  echo "Error generating project files"
  popd
  exit /b 1
)
popd
echo "Generated build/MikanMediaPipe.sln"
exit /b 0
