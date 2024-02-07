# Git简介



* 参考链接：[Git简介与简单指令](https://zhuanlan.zhihu.com/p/99313784)

* git是什么：git是一个项目管理软件，可完成对较小项目的管理。在一个需要多人合作完成的项目当中，git可记录项目文件中不同人的提交记录，提交内容与不同时间点的备份。

* 主要作用：能够更方便我们**管理这些不同版本的项目文件**，git提供了了版本控制器。所有修改后的信息会被管理在其生成的.git文件夹之中

  



# Git的使用



* git可以进行版本控制，可以保存各个版本的信息与代码

* 分类：

  1.本地版本控制：个人使用，可记录代码，文档等

  2.集中式版本控制（代表：svn）：将所有信息放于一个统一的中央服务器之上，进行统一的管理有权限控制与个人账户等等。

  3.分布式版本控制（代表：Git）：在每台电脑上均有自身的版本控制系统，每个人均拥有全部代码，可在本地访问所有的历史记录，联网之后可以放到中央服务器上。风险更小，但可能不够安全。

* git相关的Linux指令：

  ```linux命令行
  cd :      #改变目录
  cd ..     #返回上一级目录
  pwd       #显示当前所在文件目录
  is(ll) :  #列出当前目录所有文件
  touch :   #新建一个文件
  rm :      #删除一个文件
  mkdir :   #新建一个目录（文件夹）
  rm -r :   #删除一个目录
  mv        #移动文件
  reset     #初始化终端，清屏
  clear     #清屏
  history   #查看历史命令
  help      #帮助
  exit      #退出
  #         #注释
  ```

  



# GitHub基本使用



* 参考网址

  [手把手教你用git上传项目到GitHub](https://zhuanlan.zhihu.com/p/193140870)

  [如何将本地项目上传到GitHub](https://zhuanlan.zhihu.com/p/28377120)



* 本地文件上传

  1.上传项目：

  * 打开`git.bash`

  * ```cmd
    git init //把这个本地目录变成Git可以管理的仓库
    git add fileName.filetype //添加文件到仓库
    git add . //不但可以跟单一文件，还可以跟通配符，更可以跟目录。一个点就把当前目录下所有未追踪的文件全部add了 
    git commit -m "first commit" //把文件提交到仓库
    git remote add origin git@github.com:wangjiax9/practice.git //关联远程仓库，其中git@github.com:wangjiax9/practice.git为github仓库的sdd地址
    git push -u origin master //把本地库的所有内容推送到远程库上
    ```

  2.更新项目：

  * ```cmd
    git add . 
    git commit -m "first commit"  //“”内为该次提交的信息注释
    git push -u origin master
    ```

  

* 项目下载
  * 直接下载安装包
  * 通过命令行
    1. 创建本地git仓库：`git init`
    2. 克隆项目到本地：`git clone http:\\your_url`



* 项目关联
  1. 初始化本地git仓库：`git init`
  2. 挂在项目：`git remote add origin http:\\your_url` ---> 此处关联了origin（可被替换为任何名称）与URL，origin为本地仓库名称，其存储于git文件中，而非每个文件夹`.git`中，**故不同项目中可能存在名称冲突**
  3. 下载所需分支：`git pull origin master` ---> 将origin对应的URL项目中的master分支中的内容同步到本地文件夹



* 他人项目github上克隆：直接在他人项目中fork即可





#  Colab执行GitHub项目



* 打开网址：**https://colab.research.google.com/github/**，选择github，并输入github网址。打开其中后缀为ipynb的文件，按其中操作执行即可。



* 使用本地显卡运行`colabdaima`

  1. ```cmd
     jupyter serverextension enable --py jupyter_http_over_ws
     ```

  2. ```cmd
     jupyter notebook \
         --NotebookApp.allow_origin='https://colab.research.google.com' \
         --port=8848 \
         --NotebookApp.port_retries=0
     ```

  3. 复制其中本机节点到`colab`，点击连接即可





# GitHub个人博客



* 参考文献/fork项目：https://github.com/lemonchann/lemonchann.github.io

