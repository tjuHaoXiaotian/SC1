# win10 搭建星际争霸1（StarCraft I）强化学习环境

[TOC]
## 写在最开始：
```
这是一个从头写到尾的傻瓜式安装教程，内容略枯燥繁琐，请谨慎阅读。
```
寒假在家基于星际2实现micromanagement 的DRL代码，但是由于以往论文都是针对星际1，针对星际1的地图，有足够多的方法和模型间的对比数据，而星际2上的工作相对较少（基于星际2的多agent工作应该还没有），模型之间孰优孰劣不易比较，因此决定搭建星际1环境，进而比较模型胜率。

现有的星际1环境搭建需要**两台主机**，[官方推荐](https://github.com/TorchCraft/TorchCraft/blob/master/docs/user/starcraft_in_windows.md#installing-torchcraft) （1）**windows** 搭建SC:BW游戏环境作为Server端，负责游戏渲染并监听来自client端的commands输入；（2）**Linux** 基于TorchCraft，接收来自Server的环境观察（state）并发送回指令操作（actions）。server与client之间基于 ZeroMQ通信。

但是使用两台主机操作繁琐不便，因此决定将环境统一部署到同一台Windows主机（也是因为在家资源有限...），由于安装过程实在繁琐，故写下此文以备查看，在这里同时提供windows(x64)下安装环境所需的 ++.dll++ 与 ++torchcraft-1.4.0-cp35-cp35m-win_amd64.whl++ / ++torchcraft-1.4.0-cp36-cp36m-win_amd64.whl++文件下载，有需要的童鞋可以自行下载，省去自行编译过程。

## 系统环境
系统：Windows 10 x64  
CPU：CORE i7 7th  
内存：16G  
显卡：GTX 1060  

## Server 安装
### 所需文件下载
* <a name="starcraft">[startcraft.zip  [StarCraft:Brood War, version=1.16.1]](https://github.com/tjuHaoXiaotian/SC1/blob/master/install_env/server/starcraft.zip) （[iccup下载链接](https://iccup.com/starcraft/content/news/iccup_updated_to_1.16.1.html)）</a>
* <a name="BWAPI">[BWAPI](https://github.com/tjuHaoXiaotian/SC1/blob/master/install_env/server/BWAPI_420_Setup.exe) （[更多下载](https://github.com/bwapi/bwapi/releases)）</a>
* [TorchCraft ](https://github.com/TorchCraft/TorchCraft/releases)

### 安装
参考自 [Installing TorchCraft](https://github.com/TorchCraft/TorchCraft/blob/master/docs/user/starcraft_in_windows.md#torchcraft-common-prerequisites)

#### 安装 StarCraft (1.16.1)
<a href="#starcraft">下载文件（zip）</a>，解压缩到任意目录下即可，解压缩目录记为 `$STARCRAFT`。

#### 安装 BWAPI
<a href="#BWAPI">下载最新发布版本安装包</a>，安装到目录 `$STARCRAFT\BWAPI`。  
安装完成，`$STARCRAFT`目录下多出以下两个文件
* BWAPI
* bwapi-data

#### TorchCraft, common prerequistes
下载最新发布版本 [release](https://github.com/TorchCraft/TorchCraft/releases/) 
* Copy `$STARCRAFT/TorchCraft/config/bwapi.ini` in `$STARCRAFT/bwapi-data/bwapi.ini.`
* Copy `$STARCRAFT/TorchCraft/config/torchcraft.ini` in `$STARCRAFT/bwapi-data/torchcraft.ini.`
* Copy `$STARCRAFT/TorchCraft/BWEnv/bin/*.dll` into `$STARCRAFT/`
* Copy `$STARCRAFT/TorchCraft/maps/*` into `$STARCRAFT/Maps/BroodWar`

#### TorchCraft AIModule (DLL) for users:

* Extract `BWEnv.dll` from the latest archive in the [release](https://github.com/TorchCraft/TorchCraft/releases/) page and put it in `$STARCRAFT`
* Run `$STARCRAFT/BWAPI/ChaosLauncher/Chaoslauncher-MultiInstance.exe` as **administrator**.
* Check the “RELEASE” box from BWAPI.
* Click Start.  

![ChaosLauncher](https://github.com/tjuHaoXiaotian/SC1/blob/master/install_env/server/ref/ref1.png?raw=true)  

出现如下图所示情况，不用怀疑，是ok的，Server端等待Client输入，故暂时无响应。

![等待输入](https://raw.githubusercontent.com/tjuHaoXiaotian/SC1/master/install_env/server/problem/not_response.png)
---
Server端安装到此结束。

## Client 安装
```
引自官方一段话：
TorchCraft is a BWAPI module that sends StarCraft data out over a ZMQ connection. This lets you parse StarCraft data and interact with BWAPI from anywhere. The TorchCraft client should be installed from C++, Python, or Lua. We provide off the shelf solutions for Python and Lua.

可以发现，TorchCraft 并没有直接提供Windows下python扩展包，从原码包直接编译安装需要c/c++编译环境，所以相对于Linux下，Windows安装过程较为曲折。
```
### 编译安装

#### 所需文件下载（原码安装，需自行编译）
```
如果恰好你也是win10(x64, python 3.5/3.6)，可跳过编译安装，到免编译安装部分。
```
* Python 3 (我这里用的python3.5)
* Windows c/c++编译环境 [MinGW-w64](https://sourceforge.net/projects/mingw-w64/files/latest/download?source=files)
* [zstd lib](https://github.com/facebook/zstd) [压缩算法]
* [Zeromq4+](https://github.com/zeromq/zeromq4-1/releases) [Server/Client通信基础]
* Microsoft Visual Studio 2013+ （[2017下载链接](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017)）[编译 zstd/ZeroMQ]

#### 安装MinGW-w64
```
TorchCraft 用到了 c++ 11，msvs 对 c++11 支持很不友好，故安装 MinGW，但是torchcraft同时用到pthread线程库，mingw-32 不支持，所以目前选择安装 MinGW-w64。
```
* 详细安装见 [链接](https://www.cnblogs.com/findumars/p/8289454.html)  
* 记得配置环境变量 `%MINGW_64_HOME%\bin`

#### 安装 Microsoft Visual Studio 2017
下载安装即可

#### 编译 Zeromq
* TorchCraft编译安装，只用到 libzmq.dll和 zmq.h
官方已给出 windows下的 [Stable Release 4.0.4 ](http://zeromq.org/distro:microsoft-windows)，自行下载安装即可。  
* TorchCraft的发行版zip包中，bin目录下有 libzmq.dll。
###### 自行编译安装详见以下两篇博客
* [win10下Visual Studio 2015，C++ x64编译zmq](https://www.cnblogs.com/MrOuqs/p/5801333.html)
* [Visual Studio 2015下编译zmq项目下其他项目踩进的项目引用坑](https://www.cnblogs.com/MrOuqs/p/5812040.html)
---
#### 编译 zstd
* TorchCraft编译安装，只用到了 libzstd.dll 和 zstd.h。
* 官方已给出 windows下的发行版本 [releases](https://github.com/facebook/zstd/releases)，自行下载安装即可。
###### 自行编译安装详见 zstd github
* [github 链接](https://github.com/facebook/zstd#visual-studio-windows)
---
#### 修改 python distutils 默认编译器配置为 MinGW-w64
* 目录：`%Conda_home%\envs\XXX\Lib\distutils` （这里用的Conda）
* 新建文件 `distutils.cfg`  
  文件内容：
  ```
  [build]
  compiler = mingw32
  ```
* 编辑 `cygwinccompiler.py`
  修改 `class Mingw32CCompiler(CygwinCCompiler)`
  ```
    # 修改 1: 
    if sys.maxsize == 2**31 - 1:
        ms_win=' -DMS_WIN32'
    else:
        ms_win=' -DMS_WIN64'
    self.linker_dll='g++'			
    self.set_executables(compiler='g++ -O -Wall'+ms_win,
                 compiler_so='g++ -mdll -O -Wall'+ms_win,
                 compiler_cxx='g++ -O -Wall'+ms_win,
                 linker_exe='g++',
                 linker_so='%s %s %s' % (self.linker_dll, shared_option,
                               entry_point))
                               
   # 修改 2: 
   # Include the appropriate MSVC runtime library if Python was built
   # with MSVC 7.0 or later.
   # self.dll_libraries = get_msvcr()
  ```
#### 编译安装 TorchCraft
* 记TorchCraft源码目录为 `$TorchCraft`
* 修改 `%Conda_home%\envs\XXX\include\Python.h`，在最开始加入 `#include "math.h" `，否则会出现：
  ```
   error: '::hypot' has not been declared
  ```
* 拷贝 `zmq.h`，`zstd.h` 到 `$TorchCraft\include\`
* 拷贝 `libzmq.dll`，`libzstd.dll` 到 `%Conda_home%\envs\XXX\libs`
* 如果是 `git clone`下载的源码包，Remember to init submodules: `git submodule update --init --recursive`，下载 client模块
* 可能需要更新一下 steuptools 与 pip： `python -m pip install -U pip setuptools`
* cd `$TorchCraft`, Python setup: `pip install pybind11 && pip install .`  
  出现以下则安装成功：
  ![image](https://github.com/tjuHaoXiaotian/SC1/blob/master/install_env/client/ref/success.png?raw=true)
* 最后，在python中使用torchcraft前，需要将 `libzmq.dll`，`libzstd.dll` 拷贝到 `%Conda_home%\envs\XXX\Lib\site-packages\`
 ![成功](https://github.com/tjuHaoXiaotian/SC1/blob/master/install_env/client/ref/success2.png?raw=true)

### 免编译安装
#### 所需文件下载（无需编译）
* 安装 MinGW-w64 c/c++ 环境（见上）
* ZeroMQ
    * [libzmq.dll](https://github.com/tjuHaoXiaotian/SC1/blob/master/install_env/client/libzmq.dll)
* ZSTD
    * [libzstd.dll](https://github.com/tjuHaoXiaotian/SC1/blob/master/install_env/client/libzstd.dll)
* TorchCraft
    * [torchcraft-1.4.0-cp35-cp35m-win_amd64.whl  [for python3.5]](https://github.com/tjuHaoXiaotian/SC1/blob/master/install_env/client/torchcraft-1.4.0-cp35-cp35m-win_amd64.whl)
    * [torchcraft-1.4.0-cp36-cp36m-win_amd64.whl [for python3.6]](https://github.com/tjuHaoXiaotian/SC1/blob/master/install_env/client/torchcraft-1.4.0-cp36-cp36m-win_amd64.whl)
#### 简单安装
* 将 `libzmq.dll`，`libzstd.dll` 拷贝到 `%Conda_home%\envs\XXX\Lib\site-packages\`
* `pip install path_to_torchcraft-1.4.0-cpxx-cpxx-win_amd64.whl`
![image](https://github.com/tjuHaoXiaotian/SC1/blob/master/install_env/client/ref/suc3.png?raw=true)
---
Client端安装到此结束。


## 简单运行效果
* `cd $TorchCraft` (TorchCraft 源码目录)
* `cd examples/`
* `python py/example.py -t 127.0.0.1`

![image](https://github.com/tjuHaoXiaotian/SC1/blob/master/install_env/client/ref/success_end.png?raw=true)
![image](https://github.com/tjuHaoXiaotian/SC1/blob/master/install_env/client/ref/success_end2.png?raw=true)