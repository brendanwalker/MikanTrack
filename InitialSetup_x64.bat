@echo off
setlocal

:: MikanMediaPipe dependency fetcher
:: Modeled on MikanXR's InitialSetup_x64.bat (D:\Github\git-BrendanWalker\MikanXR)
:: Downloads prebuilt dependencies into deps/ and ML models into models/

set UNZIP_EXE=%~dp0tools\7zip\7za.exe

:: Make sure git submodules are present (imgui, glm, nlohmann_json, readerwriterqueue)
echo "Updating git submodules..."
git submodule update --init --recursive
IF %ERRORLEVEL% NEQ 0 (
  echo "Error updating git submodules"
  goto failure
)

:: Clean up the old deps folder
IF EXIST deps (
del /f /s /q deps > nul
rmdir /s /q deps
)

mkdir deps
pushd deps

:: ---------------------------------------------------------------- SDL2
echo "Downloading SDL2 2.30.10 (devel VC)..."
curl -L https://github.com/libsdl-org/SDL/releases/download/release-2.30.10/SDL2-devel-2.30.10-VC.zip --output sdl2-devel.zip
IF %ERRORLEVEL% NEQ 0 (
  echo "Error downloading SDL2-devel-2.30.10-VC.zip"
  goto failure
)
%UNZIP_EXE% x sdl2-devel.zip -y > nul
IF %ERRORLEVEL% NEQ 0 (
  echo "Error unzipping SDL2-devel-2.30.10-VC.zip"
  goto failure
)
del sdl2-devel.zip

:: ---------------------------------------------------------------- GLEW
echo "Downloading GLEW 2.2.0..."
curl -L https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0-win32.zip --output glew.zip
IF %ERRORLEVEL% NEQ 0 (
  echo "Error downloading glew-2.2.0-win32.zip"
  goto failure
)
%UNZIP_EXE% x glew.zip -y > nul
IF %ERRORLEVEL% NEQ 0 (
  echo "Error unzipping glew-2.2.0-win32.zip"
  goto failure
)
del glew.zip

:: ---------------------------------------------------------------- OpenCV 4.10.0
:: Same package MikanXR uses; extracted with 7za rather than run as an installer.
echo "Downloading OpenCV 4.10.0 (this one is large, ~900MB)..."
curl -L https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-windows.exe --output opencv-4.10.0-windows.exe
IF %ERRORLEVEL% NEQ 0 (
  echo "Error downloading opencv-4.10.0-windows.exe"
  goto failure
)
%UNZIP_EXE% x opencv-4.10.0-windows.exe -y > nul
IF %ERRORLEVEL% NEQ 0 (
  echo "Error extracting opencv-4.10.0-windows.exe"
  goto failure
)
del opencv-4.10.0-windows.exe

:: ---------------------------------------------------------------- ONNX Runtime (DirectML flavor)
:: A .nupkg is a zip. Contains headers + onnxruntime.dll built against DirectML.
echo "Downloading ONNX Runtime DirectML 1.20.1..."
curl -L https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/1.20.1/microsoft.ml.onnxruntime.directml.1.20.1.nupkg --output onnxruntime-directml.nupkg
IF %ERRORLEVEL% NEQ 0 (
  echo "Error downloading Microsoft.ML.OnnxRuntime.DirectML 1.20.1"
  goto failure
)
%UNZIP_EXE% x onnxruntime-directml.nupkg -oonnxruntime -y > nul
IF %ERRORLEVEL% NEQ 0 (
  echo "Error unzipping onnxruntime-directml.nupkg"
  goto failure
)
del onnxruntime-directml.nupkg

echo "Downloading DirectML 1.15.4..."
curl -L https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg --output directml.nupkg
IF %ERRORLEVEL% NEQ 0 (
  echo "Error downloading Microsoft.AI.DirectML 1.15.4"
  goto failure
)
%UNZIP_EXE% x directml.nupkg -odirectml -y > nul
IF %ERRORLEVEL% NEQ 0 (
  echo "Error unzipping directml.nupkg"
  goto failure
)
del directml.nupkg

popd

:: ---------------------------------------------------------------- ML models
:: MediaPipe models converted to ONNX, hosted by the OpenCV Zoo project.
:: https://github.com/opencv/opencv_zoo (Apache-2.0)
IF NOT EXIST models mkdir models
pushd models

echo "Downloading palm detection model..."
curl -L https://huggingface.co/opencv/opencv_zoo/resolve/main/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx --output palm_detection.onnx
IF %ERRORLEVEL% NEQ 0 goto model_failure

echo "Downloading hand landmark model..."
curl -L https://huggingface.co/opencv/opencv_zoo/resolve/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx --output hand_landmark.onnx
IF %ERRORLEVEL% NEQ 0 goto model_failure

popd

echo "Initial setup complete!"
goto exit

:model_failure
popd
echo "Error downloading a model file"
goto failure

:failure
echo "Setup failed"
exit /b 1

:exit
exit /b 0
