# 离散平均曲率和高斯曲率计算

## 实验框架逻辑变化

现在已经基本实现将OpenUSD stage与polyscope同步的逻辑。当前`Stage Viewer`窗口中的项目可以直接显示在`Polyscope Renderer`中，而不需要使用`write_polyscope`节点。因此，`Stage Viewer`中的`Import`方法已经可用了，可以直接导入OpenUSD格式的场景文件，`Polyscope Renderer`中会显示所有`Mesh` `Points` `BasisCurves`类型的几何体。

具体来说，以读取`obj`文件为例，`load_obj_pxr`节点和`write_usd`节点会将模型数据写入到`USD stage`中，`read_usd`节点会从`USD stage`中读取模型数据。由于此时模型已经存储在`USD stage`中，所以可以删去`write_usd`节点的连接，`read_usd`节点依然可以读取到模型，**而且关闭程序后再次打开，模型依然存在**。

## 实验步骤

### 1. Git 拉取仓库更新

对于一般的Git仓库，只需执行`git pull`命令，即可拉取仓库最新的更新。

然而，本实验框架仓库是包含子模块(`git submodule`)的目录，执行`git pull`后，只会更新直接包含在仓库中的文件，而不会更新仓库的子模块。

如果要确保仓库中所有内容都最新，需要在拉取后，再更新子模块。命令如下：

```shell
git pull    # 拉取仓库更新
git submodule update --init --recursive    # 更新子模块
```

这里提供一种更方便的方式。以下命令可以为给git添加一个新命令`git pullall`，执行该命令时，会自动拉取仓库更新并更新子模块。

```shell
git config --global alias.pullall '!f(){ git pull "$@" && git submodule update --init --recursive; }; f'
```

执行完上述命令后，只需执行

```shell
git pullall
```

即可拉取仓库更新并更新子模块。

### 2. 编译运行

拉取完成后，使用CMake配置项目的同学，最好重新执行一次CMake，以确保新添加的文件被正确添加到项目中。

使用VS文件夹模式打开项目的同学，可以直接重新用VS打开，VS会自动重新进行CMake配置。

打开VS之后，按下`Ctrl+Shift+B`完整编译项目，然后按下`F5`运行项目即可。

万一出现无法正确运行的问题，尝试“生成”-“重新生成解决方案”/“全部重新生成”。

### 2. 实现离散平均曲率和高斯曲率计算

完成`source/Editor/geometry_nodes/node_curvature.cpp`中的`compute_mean_curvature`和`compute_gaussian_curvature`函数，实现网格上的平均曲率和高斯曲率计算。

其中网格数据结构为OpenMesh的`PolyMesh_ArrayKernelT`，`pxr::VtArray<float>`是OpenUSD提供的数组类型，可以当作`std::vector<float>`使用。

你需要将每个顶点的曲率按照顶点索引的顺序填入`shortest_path_vertex_indices`中，无需返回值。

```cpp
typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

void compute_mean_curvature(
    const MyMesh& omesh,
    pxr::VtArray<float>& mean_curvature)
{
    // TODO: Implement the mean curvature computation
    //  You need to fill in `mean_curvature`
}

void compute_gaussian_curvature(
    const MyMesh& omesh,
    pxr::VtArray<float>& gaussian_curvature)
{
    // TODO: Implement the Gaussian curvature computation
    //  You need to fill in `gaussian_curvature`
}
```

### 3. 测试曲率计算

将`assignment/assignment2`目录下的`satge.usdc`文件放到`Assets`目录下，然后运行程序，可以看到`Stage Viewer`窗口中出现了`mesh_0`项，右击`mesh_0`项，选择`Edit`打开节点编辑器后，所有节点自动由下向上计算，将计算结果用作为一个`Polyscope`的`Scalar Quantity`绑定在网格的顶点上。你可以在`Polyscope Structure Info`窗口中找到`/mesh_0`网格的，启用`mean_curvature`或`gaussian_curvature`查看计算结果。

节点编辑器中，不同颜色的节点分别代表：

-   蓝色的节点为执行成功的节点

-   黄色的节点为希望执行，但输入不全，因此没有执行的节点

-   红色的节点为执行失败的节点

-   无色的节点为不希望执行的节点

所以如果没有正常显示模型，或没有正确可视化时，可以通过观察节点的颜色得知发生错误的位置。例如，如果`load_obj_pxr`节点为红色，说明读取模型文件失败，可能是路径错误或文件不存在。

节点编辑器会从希望执行的节点开始，向前寻找所有需要执行的节点，然后从前向后执行，并将输出传递给下一个节点。

如果你想要编写节点，可以参考已经存在的节点文件的格式，首先需要包含`nodes/core/def/node_def.hpp`，然后分别使用以下的宏定义节点的输入输出、执行函数、是否希望执行等。

```cpp
// 开始定义节点
NODE_DEF_OPEN_SCOPE

// 定义节点的输入输出
NODE_DECLARATION_FUNCTION(name_of_the_node)
{
    b.add_input<type>("input_name");
    // ...
    b.add_output<type>("output_name");
    // ...
}

// 定义节点的执行函数
NODE_EXECUTION_FUNCTION(name_of_the_node)
{
    // 当节点执行成功时，返回true以告知节点系统；否则返回false，节点系统将终止执行
    return true;
}

// 若希望节点执行，则加上这个宏
NODE_DECLARATION_REQUIRED(name_of_the_node)

// 定义节点的显示名称
NODE_DECLARATION_UI(name_of_the_node)

// 结束定义节点
NODE_DEF_CLOSE_SCOPE
```

测试成功后，可以更换其他模型测试，例如`Binaries/Debug/resources/Geometry/`目录下有一些模型文件。修改节点`load_obj_pxr`的参数为模型文件的路径，连接`write_usd`节点即可。由于此时模型已经存储在`USD stage`中，所以可以删去`write_usd`节点的连接，`read_usd`节点依然可以读取到模型，而且关闭程序后再次打开，模型依然存在。

## 实验提交

将`node_curvature.cpp`打包为 `zip` 文件，并将其命名为 `学号_姓名_hw2.zip`，通过邮件发送至 `hwc20040629@mail.ustc.edu.cn`，在邮件主题中注明课程名称、作业序号和学号、姓名。

## 除作业内容外，实验框架的更新

1.  （应该）修复了无法使用中文注释的问题

2.  修复了在VS中右击`Edit`后，程序报错的问题
    
    实际上，原因是在VS中，调试运行时，程序的执行路径并非对应的`exe`文件的路径，而是基于CMake的`build`路径下的某一个路径，所以无法找到节点的`json`文件

3.  修复了`read_obj`节点中，读取路径需要再加一个`../`的问题，原因同上