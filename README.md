# AutoStash工程

这个工程将PPCG及其依赖的ISL、PET等库配置为第三方依赖库，并编译成静态库，以便在其他项目中使用。

## 依赖项

- PPCG
- CMake (>= 3.10)
- GMP库
- MPFR库
- C++17兼容的编译器
- OpenMP (可选)
- OpenCL (可选)

### 安装依赖项

在Ubuntu/Debian系统上：
```bash
sudo apt-get install libgmp-dev libmpfr-dev
```

在CentOS/RHEL系统上：
```bash
sudo yum install gmp-devel mpfr-devel
```

在macOS上（使用Homebrew）：
```bash
brew install gmp mpfr
```

如果使用Conda环境：
```bash
conda install -c conda-forge gmp mpfr
```

## 目录结构

```
AutoStash/
├── CMakeLists.txt                    # 主CMake配置文件
├── main.cpp                          # 示例应用程序
└── ppcg/                         # PPCG及其依赖库
    ├── isl/                          # ISL库 (子模块)
    │   └── imath/                    # IMath库 (ISL子模块)
    └── pet/                          # PET库 (子模块)
```

## 如何构建

1. 克隆仓库并初始化子模块

```bash
git clone <仓库地址>
cd AutoStash
cd ppcg
./get_submodules.sh
cd ..
```

2. 创建构建目录并编译

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

3. 安装库（可选）

```bash
sudo make install
```

## 使用示例

示例程序展示了如何使用PPCG库来优化代码：

```bash
./AutoStash input.c output.c
```

## 配置选项

CMake配置中提供以下选项：

- `BUILD_SHARED_LIBS`: 是否构建共享库（默认：OFF）
- `USE_OPENMP`: 启用OpenMP支持（默认：ON）
- `USE_OPENCL`: 启用OpenCL支持（默认：OFF）

可以通过以下方式修改：

```bash
cmake -DBUILD_SHARED_LIBS=ON -DUSE_OPENCL=ON ..
```

如果CMake找不到GMP或MPFR库，可以手动指定它们的位置：

```bash
cmake -DGMP_LIBRARIES=/path/to/libgmp.so -DGMP_INCLUDE_DIRS=/path/to/gmp/include -DMPFR_LIBRARIES=/path/to/libmpfr.so -DMPFR_INCLUDE_DIRS=/path/to/mpfr/include ..
```

对于conda环境，可以使用：

```bash
cmake -DGMP_LIBRARIES=$CONDA_PREFIX/lib/libgmp.so -DGMP_INCLUDE_DIRS=$CONDA_PREFIX/include -DMPFR_LIBRARIES=$CONDA_PREFIX/lib/libmpfr.so -DMPFR_INCLUDE_DIRS=$CONDA_PREFIX/include ..
```

## 注意事项

- 与原始的PPCG库相比，此工程使用CMake来构建项目，而不是Autotools
- 如果遇到编译错误，可能需要检查子模块是否与PPCG版本匹配
- 本项目不会在第三方库中创建新文件，仅通过CMake配置来构建它们 