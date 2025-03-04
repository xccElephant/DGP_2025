# 网格上的最短路径计算

## 实验框架原理简介

实验框架的主要组成部分包括：

- 节点编辑器：用于可视化编辑程序运行逻辑。每当节点的连接结构或参数发生变化时，所有的节点都会自底向上地重新计算。

- 可视化界面：框架包含两套可视化框架，本课程仅使用基于[Polyscope](https://polyscope.run/)的可视化框架。

两部分之间有一定的联动，例如当在可视化界面中选中顶点/面/边时，所有的节点会重新计算。

## 实验步骤

### 1. 配置实验框架

按照仓库根目录的[配置方法](../../README.md)配置实验框架。

### 2. 实现最短路径计算

完成`source/Editor/geometry_nodes/node_shortest_path.cpp`中的`find_shortest_path`函数，实现网格上的最短路径计算。其中网格数据结构为OpenMesh的`PolyMesh_ArrayKernelT`。

你需要将最短路径的顶点索引填入`shortest_path_vertex_indices`中，并将最短路径的长度填入`distance`中。如果最短路径不存在，返回`false`，否则返回`true`。

```cpp
typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

// Return true if the shortest path exists, and fill in the shortest path
// vertices and the distance. Otherwise, return false.
bool find_shortest_path(
    const MyMesh::VertexHandle& start_vertex_handle,
    const MyMesh::VertexHandle& end_vertex_handle,
    const MyMesh& omesh,
    std::list<size_t>& shortest_path_vertex_indices,
    float& distance)
{
    // TODO: Implement the shortest path algorithm
    // You need to fill in `shortest_path_vertex_indices` and `distance`

    return false;
}
```

### 3. 测试最短路径计算

将`assignment/assignment1`目录下的`satge.usdc`文件放到`Assets`目录下，然后运行程序，可以看到`Stage Viewer`窗口中出现了`mesh_0`项，右击`mesh_0`项，选择`Edit`打开节点编辑器后，所有节点自动由下向上计算。具体的逻辑可以观察节点编辑器。

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


选点方法：

1. 首先在`Polyscope Structure Info`窗口中打开`/mesh_0`网格的`Edges`选项，显示网格的边，方便点击选取顶点。

2. 在`Polyscope Renderer`窗口中ctrl+左键点击选取起点，左键点击选取终点。点击选取的顶点/面信息会显示在`Polyscope Picking Info`窗口中，所以可以通过观察该窗口来确认是否选中了顶点。

选好两个顶点之后，最短路径将通过一条折线显示在网格上。

测试成功后，可以更换其他模型测试，例如`Binaries/Debug/resources/Geometry/`目录下有一些模型文件。修改节点`load_obj_pxr`的参数为模型文件的路径即可读取。

## 实验目的

熟悉网格数据结构

## 实验提交

将`node_shortest_path.cpp`打包为 `zip` 文件，并将其命名为 `学号_姓名_hw1.zip`，通过邮件发送至 `hwc20040629@mail.ustc.edu.cn`，在邮件主题中注明课程名称、作业序号和学号、姓名。