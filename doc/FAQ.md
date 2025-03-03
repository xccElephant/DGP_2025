# 常见问题

## 用方法1配置SDK时出现错误/无法下载/配置速度慢/……

建议使用方法2

## 配置好SDK后，使用VS生成时出错

1.  检查子模块是否全部正确下载

    在框架根目录下打开终端，输入

    ```shell
    git submodule update --init --recursive
    ```

    直到没有任何输出为止。

2.  检查CMake是否正确识别SDK中的OpenUSD

    若正确识别，在CMake的输出中可以找到
    
    ```
    ...
    CMAKE_BUILD_TYPE is Debug. Selecting SDK from Debug
    Found Vulkan version 1.3.296
    Found OpenUSD version
    ...
    ```
        
    若未正确识别，在CMake的输出中可以找到
    
    ```
    CMake Warning at CMakeLists.txt:99 (find_package): 
    By not providing "Findpxr.cmake" in CMAKE_MODULE_PATH this project has asked CMake to find a package configuration file provided by "pxr", but CMake did not find one.

    Could not find a package configuration file provided by "pxr" with any of the following names:

    pxrConfig.cmake
    pxr-config.cmake

    Add the installation prefix of "pxr" to CMAKE_PREFIX_PATH or set "pxr_DIR" to a directory containing one of the above files. If "pxr" provides a separate development package or SDK, be sure it has been installed.
    ```

    请检查`DGP_2025/SDK/OpenUSD/Debug/`目录下是否有`pxrConfig.cmake`文件，若没有，请重新配置SDK。

    如果有，请检查`CMAKE_BUILD_TYPE`是否正确，如构建类型为`Debug`时，CMake输出中应包含

    ```
    CMAKE_BUILD_TYPE is Debug. Selecting SDK from Debug
    ```

3.  尽量使用VS提供的全部生成/生成解决方案功能，而不是单独生成某个项目，避免出现缺乏依赖的情况。

## 运行框架时，点击`Edit`后报错

使用全部生成/生成解决方案功能，而不是单独生成某个项目。