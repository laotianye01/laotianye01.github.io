# 大学数学



## 有关数学体系的说明

```
         【公理】           ← 不可证明，体系起点
           ↓
         【定义】           ← 不是命题，无真假
           ↓
         【公例】           ← 极基础结论（可有简单证明）
           ↓
         【定理】           ← 有完整证明，可由公理、公例或定理推出
           ↓
         【推论】           ← 定理的直接后果，通常不单独成立为主命题
         
```







## 线性代数

* 参考链接：
  1. [《线性代数》知识点汇总](https://zhuanlan.zhihu.com/p/457894126)
  1. 有关书(目前看到chapter3)



### **Vector** & Matrix

* Vector definition
  $$
  expression \space of \space vector \space v = \left[ \begin{array}{l} v_1 \\ ... \\ v_i \end{array} \right]
  $$

* Vector operation definition
  $$
  assume \space v = \left[ \begin{array}{l} v_1 \\ v_2 \end{array} \right],w = \left[ \begin{array}{l} w_1 \\ w_2 \end{array} \right] \\
  VECTOR\space ADDITION: v+w = \left[ \begin{array}{l} v_1+w_1 \\ v_2+w_2 \end{array} \right] \\
  SCALAR\space MULTIPLICATION: cv=\left[ \begin{array}{l} cv_1\\ cv_2 \end{array} \right]\\
  $$

* Linear combination definition: 对于二维空间的情况，当v, w两个向量不共线时，用v, w的线性组合可以表示二维空间内的全部向量，拓展到n维空间也同理
  $$
  LINEAR\space COMBINATION:cv + dw=\left[ \begin{array}{l} cv_1+dw_1 \\ cv_2+dw_2\end{array} \right]
  $$

* Dot Product / inner product && length definition
  $$
  for \space v = (v1,v2) \space and \space w = (w1,w2) \\
  dot \space product:v \cdot w = v_1w_1 + v_2w_2 \\ 
  length \space of \space v= \|\vec{v}\| = \sqrt{\vec{v} \cdot \vec{v}} = \left(v_1^2 + v_2^2 + \cdots + v_n^2\right)^{1/2}
  $$

* 向量夹角(定理) -- 可通过向量分解证明
  $$
  Unit\space vectors\space u\space and\space U\space have\space include\space angle\space \theta\rightarrow  u \cdot U = \cos \theta \\
  \frac{\vec{v} \cdot \vec{w}}{\|\vec{v}\| \|\vec{w}\|} = \cos \theta
  $$

* vector independent & dependent(以三维矩阵为例)

  * invertible matrix -- 矩阵中列向量u, v, w independent -- 任意列向量不可由其他向量表示 -- Ax = 0有唯一解
  * singular matrix -- 矩阵中列向量u, v, w dependent -- 存在列向量可由其他向量表示 -- Ax = 0有多个解



**线性组合 -- 方程组 -- 矩阵**

* 以二维空间为例，假设我希望使用不共线的向量x1, x2表示任意一个向量k，可通过如下方式求解
  $$
  cx_1 + dx_2 = b \leftrightarrow \left[ \begin{array}{l} b_1 \\ b_2\end{array} \right] =\left[ \begin{array}{l} cx_{11}+dx_{21} \\ cx_{12}+dx_{22}\end{array} \right] \\ 
  \leftrightarrow \\
  该过程等价于求解以下二元方程组(c,d为未知数)的解\\ 
  \leftrightarrow \\
  \left\{\begin{aligned}
  b_1=cx_{11}+dx_{21} \\
  b_2=cx_{12}+dx_{22}
  \end{aligned}\right. \\ 
  \leftrightarrow \\
  为了简化以上求解过程，我们便可以引入矩阵的概念并定义矩阵乘法 \\
  \leftrightarrow \\
  C = \left[ \begin{array}{l} x_{11} & x_{21} \\ x_{12} & x_{22}\end{array} \right], x = \left[ \begin{array}{l} c \\ d\end{array} \right],b=\left[ \begin{array}{l} b_1 \\ b_2\end{array} \right] \\
  Cx=b\leftrightarrow \\
  其中Cx部分也可以看作为向量的点乘 \\
  \leftrightarrow \\
  Cx = \left[ \begin{array}{l} x_{11} & x_{21} \\ x_{12} & x_{22}\end{array} \right] \left[ \begin{array}{l} c \\ d\end{array} \right] = \begin{bmatrix} (x_{11}, x_{21}) \cdot (c, d) \\ (x_{12}, x_{22}) \cdot (c, d) \end{bmatrix}
  $$

* 方程组(矩阵)求解 -- 消元法

  * 高斯消元法

    <img src="https://s2.loli.net/2024/05/13/VRKe3iHJWcw1vFy.png" alt="image-20240513193229454" style="zoom:50%;" />

    1. 先将第n行的前(n-1)个元素消掉 (使用类似迭代的方法，先使用第一行将其余行的第一个元素消去，再使用第二行将全部的第二个元素消去，以此类推，得到一个三角形的矩阵->不要忘记值)
    2. 再由下向上消元得到对角阵 (使用最后一行$x_n$带入上一行得到$x_{n-1}$​的值，依此类推)

  * Invert Matrix $Ax=b \leftrightarrow x = A^{-1}b$ -- 表示方程组有解 -- 表示向量b可用A矩阵中列向量表示
    <img src="https://cdn.jsdelivr.net/gh/laotianye01/img_bed@image/note_image/image-20250710151528005.png" alt="image-20250710151528005" style="zoom: 33%;" />

  * 消元的过程可通过矩阵乘法(由消元过程引出)实现
    $$
    example:三维矩阵中从第二行减去两倍第一行 \\
    E(变换矩阵) = \begin{bmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}, C(原矩阵)=\begin{bmatrix} 2 \\ 8 \\ 10 \end{bmatrix} \\
    \begin{bmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 2 \\ 8 \\ 10 \end{bmatrix} = \begin{bmatrix} 2 \\ 4 \\ 10 \end{bmatrix} \\ \leftrightarrow将原矩阵每个元素转换为一个行向量\leftrightarrow \\ b_i\begin{bmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 - 2b_1 \\ b_3 \end{bmatrix}
    $$

    * 理解: C'第一行为$(1,0,0)\cdot (b_1, b_2, b_3)$，结果$b_1$，第二行为$(-2,1,0)\cdot (b_1, b_2, b_3)$，结果为$-2b_1+b_2$，第三行类似(引出矩阵有关定义与运算性质)

* 引入矩阵(用于描述上述系统 -- 方程组求解可用矩阵运算解决，但矩阵运算不仅限于方程求解)

  * 矩阵基本运算
    $$
    A=[a_{ij}]_{m*n}, \space B=[b_{ij}]_{m*n}
    $$


    * 若对$\forall i,j$，如果$a_{ij}=-b_{ij}$，B=-A
    * A+B为同尺寸矩阵中每项相加
    * kA为$ka_{ij}$
    * 加法满足交换律与结合律

  * nullspace N(A): consists of all solutions to Ax= 0

  * 矩阵乘法定义 
    $$
    A=[a_{ij}]_{m*n}=\left(
    \begin{array}{l}
    a_{11} & a_{12} & ... & a_{1n} \\
    a_{21} & a_{22} & ... & a_{1n} \\
    ... & ... & ... & ... \\
    a_{m1} & a_{m2} & ... & a_{mn}
    \end{array}
    \right),\space 
    B=[b_{ij}]_{n*p}=\left(
    \begin{array}{l}
    b_{11} & b_{12} & ... & b_{1p} \\
    b_{21} & b_{22} & ... & b_{1p} \\
    ... & ... & ... & ... \\
    b_{n1} & b_{n2} & ... & b_{np}
    \end{array}
    \right)
    \\ \\
    
    (m \times n)(n \times p) = (m \times p) \rightarrow
    
    
    \begin{bmatrix} m \text{ rows} \\ n \text{ columns} \end{bmatrix} \begin{bmatrix} n \text{ rows} \\ p \text{ columns} \end{bmatrix} = \begin{bmatrix} m \text{ rows} \\ p \text{ columns} \end{bmatrix}
    
    \newline\newline
    AB=[a_{ij}]_{m*n}[b_{ij}]_{n*p}=[c_{ij}]_{m*p}\space,where \newline\newline
    [c_{ij}]_{m*p}=\sum_k^na_{ix}b_{xj}
    $$

  * 矩阵乘法运算律(定理)

    * (AB)C = A(BC)
    * A(B+C) = AB + AC
    * $(\lambda A)B = A(\lambda B) = \lambda(AB)$
    * **矩阵运算不满足交换律**，若AB = BA，则A与B是可变换的（单位阵与所有矩阵可变换）

  * 矩阵的分块成法(以下为特殊分块方式，其余方式也同理)

    * 运算定理
      $$
      A = \left(
      \begin{array}{ccc:c}
      a_{11} & a_{12} & a_{13} & a_{14} \\
      a_{21} & a_{22} & a_{23} & a_{24} \\
      \hdashline
      a_{31} & a_{32} & a_{33} & a_{34}
      \end{array}
      \right) = \left(
      \begin{array}{ccc:c}
      A_{11} & A_{12} \\
      A_{21} & A_{22} \\
      \end{array}
      \right)
      $$

    * 例子
      $$
      A=[a_{ij}]_{m*n}=\left(
      \begin{array}{l}
      a_{11} & a_{12} & ... & a_{1n} \\
      a_{21} & a_{22} & ... & a_{2n} \\
      ... & ... & ... & ... \\
      a_{m1} & a_{m2} & ... & a_{mn}
      \end{array}
      \right),\space 
      B=[b_{ij}]_{n*p}=\left(
      \begin{array}{l}
      b_{11} & b_{12} & ... & b_{1p} \\
      b_{21} & b_{22} & ... & b_{2p} \\
      ... & ... & ... & ... \\
      b_{n1} & b_{n2} & ... & b_{np}
      \end{array}
      \right) \\
      \leftrightarrow
      \\
      A(行向量表示)=[a_{i}]_{m*1}=\left(
      \begin{array}{l}
      a_{1}\\
      a_{2}\\
      ...\\
      a_{m}
      \end{array}
      \right),\space where \space a_i\in R^{1*n} ;\space\space
      B(列向量表示)=[b_{j}]_{1*p}=\left(
      \begin{array}{l}
      b_{1} & b_{2} & ... & b_{p} \\
      \end{array}
      \right),\space where \space b_j\in R^{n*1}
      \\
      AB=C=\left(
      \begin{array}{l}
      a_{1}b_{1} & a_{1}b_{2} & ... & a_{1}b_{p} \\
      a_{2}b_{1} & a_{2}b_{2} & ... & a_{1}b_{p} \\
      ... & ... & ... & ... \\
      a_{m}b_{1} & a_{m}b_{2} & ... & a_{m}b_{p}
      \end{array}
      \right)
      $$

    

* 矩阵类型

  * 方阵性质 (下为行列式)
    $$
    A = \left(
    \begin{array}{l}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23} \\
    a_{31} & a_{32} & a_{33}
    \end{array}
    \right)
    $$

    * $|A^T|=|A|$
    * $|\lambda A|=\lambda^n|A|$
    * $|AB|=|A||B|$
    * $|AB|=|BA|$

  * 对角阵 (非零全为1为单位阵E)

  $$
  A = \left(
  \begin{array}{l}
  a_{11} & 0 & ... & 0 \\
  0 & a_{22} & ... & 0 \\
  ... & ... & ... & ... \\
  0 & 0 & ... & a_{nn}
  \end{array}
  \right)
  $$

  * 增广矩阵

  $$
  A = \left(
  \begin{array}{c:c}
  \begin{matrix}
  a_{11} & a_{12} & a_{13} \\
  a_{21} & a_{22} & a_{23} \\
  a_{31} & a_{32} & a_{33}
  \end{matrix}&
  \begin{matrix}
  b_1 \\
  b_2 \\
  b_3
  \end{matrix}
  \end{array}
  \right)
  $$

* **逆矩阵**(对于方阵而言)

  * 逆矩阵表示
    $$
    1. 定义: 若AB=BA=E, \space 则B=A^{-1} \\
    2. 接上面有关方程的解部分\rightarrow 
    \left\{\begin{aligned}
    Cx=b(线性方程) \\
    x=Db(假设解存在) 
    \end{aligned}\right. \\
    \left\{\begin{aligned}
    Cx=b \\
    Cx=CDb
    \end{aligned}\right. \\ 
    (CD-I)b=0
    $$

  * 逆矩阵定理

    * 充要条件: A可逆 <--> |A| != 0
    * $(A^{-1})^{-1}=A$
    * $(kA)^{-1}=k^{-1}A^{-1}$
    * $(A^T)^{-1}=(k^{-1})^T$​
    * $(AB)^{-1}=B^{-1}A^{-1}$​

  * 矩阵求逆
    $$
    将A^{-1}看作所需求解的内容: AA^{-1}=I\rightarrow A(x_1,x_2,x_3)=I \\
    三组解对应三个列向量e_1=\left(\begin{array}{l}1 \\0 \\0\end{array}\right),e_2=\left(\begin{array}{l}0 \\1 \\0\end{array}\right),e_3=\left(\begin{array}{l}0 \\0 \\1\end{array}\right) 
    \\ \\ \leftrightarrow
    使用高斯消元法+增广矩阵，同时操作对应三组数据，通过将左侧矩阵转化为单位矩阵以求得三组解 
    \leftrightarrow \\ \\
    \left(
    \begin{array}{ccc:ccc}
    a_{11} & a_{12} & a_{13} & 1 & 0 & 0 \\
    a_{21} & a_{22} & a_{23} & 0 & 1 & 0 \\
    a_{31} & a_{32} & a_{33} & 0 & 0 & 1
    \end{array}
    \right)
    $$

  * **初等阵**: 由单位阵经过一次初等变换 (行交换，行加减，乘常数) 得到的矩阵，使用一个矩阵乘以初等阵即进行了与初等阵相同的初等变换
    $$
    E_3[2+3(4)]=\left(
    \begin{array}{l}
    1 & 0 & 0 \\
    0 & 1 & 4 \\
    0 & 0 & 1
    \end{array}
    \right)
    \newline 其等价于变换\stackrel{r_2+4r_3}{\longrightarrow}
    $$

    * 初等阵必可逆
    * 若n阶方阵P可逆，则存在有限个初等阵，$s.t.\space PE_1...E_i=E$​

  * factorization && elimination理解

    * $A = LU \leftrightarrow L^{-1}A=U;\space U为上对角矩阵 $
    * 若将上对角矩阵转化的过程表示为$L^{-1}=E_{i_1j_1}...E_{i_xj_y}$, 则可令其中变换$E_{i_xj_y}$的作用表示为A'矩阵的$A_{i_xj_y}$转化为0

* 矩阵转置

  * 定义：将A的行向量变为B的列向量
    $$
    A = \left(
    \begin{array}{l}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23}
    \end{array}
    \right), \space A^T=\left(
    \begin{array}{l}
    a_{11} & a_{21} \\
    a_{12} & a_{22} \\
    a_{13} & a_{23}
    \end{array}
    \right)
    $$

  * 有关转制定理

    * $(A^T)^T=A$
    * $(A+B)^T=A^T+B^T$
    * $(AB)^T=B^TA^T$​

  * 对称阵

    * 对称阵定义
      $$
      if \space A^T=A
      $$

      * 设C=AB，若AB均为对称阵，且AB=BA，则C也为对称阵

    * 反对称阵
      $$
      if \space A^T=-A
      $$

      * 若A为反对称阵，则$A^T$​为反对称阵



### 特征值与特征向量

* 向量基本信息

  * 定义：由n个数组成的一个数组

  * 线性表示：若一个向量$\beta$可被一个n维**向量组**{$a_1, a_2,...,a_n$}表示，即
    $$
    \beta = k_1a_1+...+k_na_n
    $$

  * 判断向量是否能被向量组线性表示 --> $aX=\beta$​有解 <---> R(A) = R(A|B) ，具体而言为增广后化简出现某一行a全零b有值则不可以

  * 基：若一个由n个n维向量组成的向量组相互间线性不相关，则其可表示该向量空间内的全部值 (将其组成矩阵A后R(A)=n, |A|!=0)

* 矩阵的秩：非零子式的最高阶数(矩阵中相互独立向量的个数)

  * 秩的求法：
    1. A 为抽象矩阵：由定义或性质求解；
    2. A 为数字矩阵：$A \rightarrow$ 初等行变换 $\rightarrow$ 阶梯型（每行第一个非零元素下面的元素均为0），则  r(A) = 非零行的行数；
  * 秩的性质：
    1. A 为 $m \times n$ 阶矩阵，则 $r(A) \leq \min(m, n)$；
    2. $r(A \pm B) \leq r(A) + r(B)$
    3. $r(AB) \leq \min\{r(A), r(B)\}$
    4. $r(kA) = r(A),( k \neq 0 )$
    5. $r(A) = r(AC), C 是一个可逆矩阵$
    6. $r(A) = r(A^T) = r(A^T A) = r(A A^T)$；
    7. 设 A 是 $m \times n$ 阶矩阵, B 是 $n \times s$ 矩阵, AB = O，则 $r(A) + r(B) \leq n$；

* 特征值：对于n阶方阵，若A为n方阵，x为列向量，$Ax=\lambda x$，则$\lambda$为A的特征值

  * kA特征值为$k\lambda$
  * f($\lambda$)为f(A)的特征值
  * $\lambda^m$为$A^m$​的特征值

* 特征值(Eigenvalues)，特征向量
  
  * 特征向量x: 对于矩阵A, 满足$Ax=\lambda x$, 即A与x相乘后x方向不变, 仅产生了一定的放缩
  
  * 特征值性质
  
    1. product of the n eigenvalues = det(A)
    2. sum of the n eigenvalues = sum of the n diagonal entries of A
  
  * 特征值求解 (解出来可能为找零点的问题)
    $$
    Ax=\lambda x \rightarrow (\lambda E-A)x=0 \\\\
    det(\lambda E-A)=\left|
    \begin{array}{l}
    \lambda - a_{11} & -a_{12} & -a_{13} \\
    -a_{21} & \lambda - a_{22} & -a_{23} \\
    -a_{31} & -a_{32} & \lambda - a_{33}
    \end{array}
    \right| = 0
    $$
  
  * 特征向量求解: 求出特征值之后带入即可
  
* 对角化
  $$
  matrix\space A\space has\space n\space linearly\space independent\space eigenvectors\space x1, ... , Xn, \space X = \begin{bmatrix} x_1 & ... & x_n  \end{bmatrix}
  \\
  可推导出\rightarrow X^{-1}AX = \Lambda = \begin{bmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{bmatrix}, where \space \lambda_i \space is \space Eigenvalue
  $$
  
* 对角化好处
  $$
  A = X\Lambda X^{-1} \\
  A^2 = (X\Lambda X^{-1})(X\Lambda X^{-1}) = X\Lambda (X^{-1}X)\Lambda X^{-1}=X\Lambda^2 X^{-1} \\
  A^n=X\Lambda^n X^{-1}
  $$

* 相似矩阵: 若存在可逆矩阵P，使$B=P^{-1}AP$，则A与B相似，记A~B 



### **行列式(Determinants)**

$$
A = \left|
\begin{array}{l}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{array}
\right|
$$

* **行列式定义式** (共有 $n!$ 项相加，$\tau$ 为逆序数)
  $$
  det(A)=\left|
  \begin{array}{l}
  a_{11} & a_{12} & ... & a_{1n} \\
  a_{21} & a_{22} & ... & a_{2n} \\
  ... & ... & ... & ... \\
  a_{n1} & a_{n2} & ... & a_{nn}
  \end{array}
  \right|= \sum_{j_1...j_n}(-1)^{\tau(j_1...j_n)}a_{1j_1}...a_{nj_n}
  $$

* 二阶矩阵行列式(一般仅低阶矩阵使用，高阶直接转换即可)
  $$
  A = \begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad-bc \\
  A = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \quad \text{has inverse} \quad A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
  $$
  
* **行列式值计算**：把高阶慢慢转到二阶，二阶的值为对角线相乘得差，每项-1的幂为其行与列的值
  $$
  det(A)=det\left[
  \begin{array}{l}
  a_{11} & a_{12} & a_{13} \\
  a_{21} & a_{22} & a_{23} \\
  a_{31} & a_{32} & a_{33}
  \end{array}
  \right]=a_{11}A_{11} + a_{12}A_{12} + a_{13}A_{13} = \newline\newline
  =a_{11}(-1)^{1+1}\left|
  \begin{array}{l}
  a_{22} & a_{23} \\
  a_{32} & a_{33}
  \end{array}
  \right| + a_{12}(-1)^{1+2}\left|
  \begin{array}{l}
  a_{21} & a_{23} \\
  a_{31} & a_{33}
  \end{array}
  \right| + a_{13}(-1)^{1+3}\left|
  \begin{array}{l}
  a_{21} & a_{22} \\
  a_{31} & a_{32}
  \end{array}
  \right| \newline\newline
  =a_{11}(a_{22}a_{33}-a_{32}a_{23}) + a_{12}(a_{21}a_{33}-a_{32}a_{23}) + a_{13}(a_{21}a_{32}-a_{31}a_{22})
  $$

* **行列式几何意义**（以二阶为例,看列向量）

  <img src="https://cdn.jsdelivr.net/gh/laotianye01/img_bed@image/note_image/image-20250710132305907.png" alt="image-20250710132305907" style="zoom:33%;" />
  $$
  det(A) = \left|
  \begin{array}{l}
  a_{11} & a_{12} \\
  a_{21} & a_{22}
  \end{array}
  \right|= a_{11}a_{22}-a_{12}a_{21} = 列向量围出平型四边形面积
  $$

  * 三阶同理，为六面体体积，依此类推。将行列式看作线性变换时，其值代表该次线性变换的缩放程度
  * **行列式为零 -- 矩阵不可逆 -- 矩阵列向量dependent**

* **行列式性质**(定理)

  1. 转置后值不变
     $$
     D_3 = \left|
     \begin{array}{l}
     a_{11} & a_{12} & a_{13} \\
     a_{21} & a_{22} & a_{23} \\
     a_{31} & a_{32} & a_{33}
     \end{array}
     \right| = D_3^T = \left|
     \begin{array}{l}
     a_{11} & a_{21} & a_{31} \\
     a_{12} & a_{22} & a_{32} \\
     a_{13} & a_{23} & a_{33}
     \end{array}
     \right|
     $$

  2. 可提出某行(列)公因子k
     $$
     D_3 = \left|
     \begin{array}{l}
     ka_{11} & ka_{12} & ka_{13} \\
     a_{21} & a_{22} & a_{23} \\
     a_{31} & a_{32} & a_{33}
     \end{array}
     \right| = k.\left|
     \begin{array}{l}
     a_{11} & a_{12} & a_{13} \\
     a_{21} & a_{22} & a_{23} \\
     a_{31} & a_{32} & a_{33}
     \end{array}
     \right|
     $$

  3. 两行(列)互换其值相反
     $$
     D = \left|
     \begin{array}{l}
     a_{11} & a_{12} & a_{13} \\
     a_{21} & a_{22} & a_{23} \\
     a_{31} & a_{32} & a_{33}
     \end{array}
     \right|=-\left|
     \begin{array}{l}
     a_{21} & a_{22} & a_{23} \\
     a_{11} & a_{12} & a_{13} \\
     a_{31} & a_{32} & a_{33}
     \end{array}
     \right|
     $$

  4. 若对应两行(列)成比例行列式为0
     $$
     D = \left|
     \begin{array}{l}
     a_{11} & a_{12} & a_{13} \\
     2a_{11} & 2a_{12} & 2a_{13} \\
     a_{31} & a_{32} & a_{33}
     \end{array}
     \right|=0
     $$

  5. 可按行(列)分解
     $$
     D = \left|
     \begin{array}{l}
     a_{11} & a_{12} & a_{13} \\
     a_{21}+b_{21} & a_{22}+b_{22} & a_{23}+b_{23} \\
     a_{31} & a_{32} & a_{33}
     \end{array}
     \right| = \left|
     \begin{array}{l}
     a_{11} & a_{12} & a_{13} \\
     a_{21} & a_{22} & a_{23} \\
     a_{31} & a_{32} & a_{33}
     \end{array}
     \right| + \left|
     \begin{array}{l}
     a_{11} & a_{12} & a_{13} \\
     b_{21} & b_{22} & b_{23} \\
     a_{31} & a_{32} & a_{33}
     \end{array}
     \right|
     $$

  6. 高斯消元行列式值不变 -- 利用性质进行行内的化简，转换为对角阵以计算(可用于对角矩阵的化简)
     $$
     D = \left|
     \begin{array}{l}
     a_{11} & a_{12} & a_{13} \\
     a_{21} & a_{22} & a_{23} \\
     a_{31} & a_{32} & a_{33}
     \end{array}
     \right|\stackrel{r_2+2r_1}{\longrightarrow}\left|
     \begin{array}{l}
     a_{11} & a_{12} & a_{13} \\
     a_{21}+2a_{11} & a_{22}+2a_{12} & a_{23}+2a_{13} \\
     a_{31} & a_{32} & a_{33}
     \end{array}
     \right|
     $$

  7. 余子式: 当某行(列)仅剩一 (多个也行，意义不大) 个不为零的值时，可按该行(列)展开行列式，得到一个 $n-1$​ 阶的行列式 (与行列式运算性质同理)
     $$
     D = \left|
     \begin{array}{l}
     0 & a_{12} & a_{13} \\
     a_{21} & a_{22} & a_{23} \\
     0 & a_{32} & a_{33}
     \end{array}
     \right| = a_{21}*(-1)^{2+1}*\left|
     \begin{array}{l}
     a_{12} & a_{13} \\
     a_{32} & a_{33}
     \end{array}
     \right|
     $$

* **行列式类型**

  * 下三角行列式(以对角线为界)
    $$
    D_n = \left|
    \begin{array}{l}
    a_{11} & 0 & ... & 0 \\
    a_{21} & a_{22} & ... & 0 \\
    ... & ... & ... & ... \\
    a_{n1} & a_{n2} & ... & a_{nn}
    \end{array}
    \right| = a_{11}a_{22}...a_{nn}
    $$

  * 副下三角行列式
    $$
    D_n = \left|
    \begin{array}{l}
    0 & 0 & ... & a_{1n} \\
    0 & 0 & ... & a_{2n} \\
    ... & ... & ... & ... \\
    a_{n1} & a_{n2} & ... & a_{nn}
    \end{array}
    \right| = (-1)^{\frac{n(n-1)}{2}}a_{1n}a_{2(n-1)}...a_{n1}
    $$

  * 范德蒙德
    $$
    D = \left|
    \begin{array}{l}
    a_1^0 & a_2^0 & ... & a_n^0 \\
    a_1^1 & a_2^1 & ... & a_n^1 \\
    ... & ... & ... & ... \\
    a_1^{n-1} & a_2^{n-1} & ... & a_n^{n-1} \\
    \end{array}
    \right| = \prod_{n>=i>=j>=1}(a_i-a_j)
    $$

  * 三线型(见链接2)

  * 箭头行(见链接2)

* 克拉默法则: 对于一组有n元n个的线性方程组，可使用其系数与结果构建n个行列式，并通过行列式求解x(见链接2)



### Linear Transformation

* definition: v is a vector, linear transformation T(v) satisfy
  $$
  T(c v + dw) = c T(v) + dT(w)
  $$
  <img src="https://cdn.jsdelivr.net/gh/laotianye01/img_bed@image/note_image/image-20250710190739451.png" alt="image-20250710190739451" style="zoom:25%;" />

* Property: all linear transformations T from $V = R^n$ to $W = R^m$ produced by matrices (denote by A in this section)
  $$
  T(v) = A(v), V\in R^n \\
  Range\space of\space T = set\space of\space all\space outputs\space T (v)\in W \\
  Kernel\space of\space T = set\space of\space all\space inputs\space for\space which\space T(v) = 0\in V\rightarrow nullspace\space of\space A
  $$

* Matrix for transformation
  $$
  for\space vector\space v\in R^n,whose\space base\space is\space(v_1,...,v_n) \\
  v=c_1v_1+c_2v_2+...+c_nv_n \rightarrow T(v)=c_1T(v_1)+c_2T(v_2)+...+c_nT(v_n)
  $$

* Find transformation matrix

  * General: $T(v)=v \rightarrow Av = w,(已知v,w求A)$

  * T and S are represented by two matrices A and B
    $$
    T(S(v))=ABv
    $$

  * Base of vector in transformation: base of $\vec{v}$ and $\vec{w}$ will result in different matrix for transformation $T(v)=w$
    $$
    Example:\space for\space transformation\space T(v)=v \\
    Input\space basis: \begin{bmatrix} v_1 & v_2 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 3 & 8 \end{bmatrix};
    \space\space
    Output\space basis: \begin{bmatrix} w_1 & w_2 \end{bmatrix} = \begin{bmatrix} 3 & 0 \\ 1 & 2 \end{bmatrix}
    \\
    \because T(v)=v, \space \therefore \begin{bmatrix} w_1 & w_2 \end{bmatrix} \begin{bmatrix} c_1 \\ c_2 \end{bmatrix} = T(\begin{bmatrix} v_1 & v_2 \end{bmatrix} \begin{bmatrix} c_1 \\ c_2 \end{bmatrix})=T(c_1v_1+c_2v_2)=c_1T(v_1)+c_2T(v_2)
    \\
    获取矩阵B，用以将输入基转化为输出基
    \rightarrow
    \left\{\begin{array}{l}
    v_1=w_1+w_2 \\
    v_2=2w_1+3w_2 
    \end{array}\right.
    \\
    \begin{bmatrix} w_1 & w_2 \end{bmatrix} \begin{bmatrix} B \end{bmatrix} = \begin{bmatrix} v_1 & v_2 \end{bmatrix} 
    \space
    \leftrightarrow
    \space
    \begin{bmatrix} 3 & 0 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 3 & 8 \end{bmatrix}
    $$

* 美化线性变换 -- 寻找合适的$base_{in}$与$base_{out}$
  $$
  线性变换满足如下性质: A'(新条件下线性变换矩阵)=X_{base_{in}}AX_{base_{out}}, \space A为in+out基为I时情况
  \\
  利用对角化知识\rightarrow \Lambda = X^{-1}AX\space(\Lambda为对角化后A,X为特征向量矩阵)
  $$
  

## 概率论

* 参考链接：[概率论复习笔记](https://zhuanlan.zhihu.com/p/445887214)



### 条件概率

* 基本知识

  * 条件概率
    $$
    P(A|B)=\frac{P(AB)}{P(B)}
    $$

  * 独立事件
    $$
    P(AB)=P(A)P(B)
    $$

  * 全概率公式
    $$
    if \space P(B_1)+P(B_2)=1 \newline\newline
    P(A)=P(AB_1)+P(AB_2)=P(B_1)P(A|B_1)+P(B_2)P(A|B_2)
    $$

  * 贝叶斯定理（使用先验概率修正后验概率）
    $$
    后验概率:P(B_i|A)=\frac{P(AB_i)}{P(A)}=\frac{P(B_i)P(A|B_i)}{\sum_i P(B_i)P(A|B_i)}
    $$

    * 雨天蓝绿车问题: 城市里有15%蓝车，85%绿车，在下雨天，20%的人会把蓝车看成绿车，则一个人在下雨天看到一车是蓝色的，那这个车是蓝车的概率是多大？

      ![a3beb462b575275749e956764fb8588](https://s2.loli.net/2024/05/14/8eM6uqXwdDk7LjP.jpg)

  

### 随机变量

* 期望 (表现x分布的中心位置)
  $$
  Ex=\sum_kx_kP_k=\int xP(x)dx
  $$

  * $E(kX+b)=kE(X)+b$
  * $E(X+Y)=E(X)+E(Y)$​
  * X,Y独立时，$E(XY)=E(X)E(Y)$

* 方差 (表现数据的波动范围大小)
  $$
  Dx=E(x-E(x))^2=\sum_k(x-E(x))^2P_k=\int (x-E(x))^2P(x)dx \newline\newline
  Dx=E(X^2)-(E(X))^2\space ,where \newline
  E(X^2)=\int x^2P(x)dx=\sum x_k^2P_k
  $$

  * $D(kX+b)=k^2D(X)$
  * X,Y独立时，$D(X+Y)=D(X)+D(Y)$

* 关联度Cov
  $$
  cov(X, Y)=E(XY)-E(X)E(Y)
  $$

* 关联系数$\rho$
  $$
  \rho_{xy} = \frac{cov(X, Y)}{\sqrt{D(X)}\sqrt{D(Y)}}
  $$
  
* 离散随机变量

  * n重二项分布x ~ B(n,p)，X为成功实验的次数 (n次实验k次成功)
    $$
    P(X=k)=C_n^kq^k(1-q)^{n-k} \space,where\newline
    E(x)=np;\space Var(x)=np(1-p)
    $$

  * 泊松分布 $x - Poi(\lambda)$​
    $$
    P(X=k)=\frac{\lambda^ke^{-\lambda}}{k!} \space,where\newline
    E(x)=Var(x)=\lambda
    $$

* 分布函数
  $$
  F(X)=P(X<=x)=\int_{-\infty}^x f(t)dt \space,where\newline\newline
  f(t)是概率密度函数p.d.f，F(X)是分布函数，丹增
  $$

* 连续随机变量

  * 均匀分布x ~ U(a, b)
    $$
    f(x)=\frac{1}{b-a} \space,where\newline
    E(x)=;\space Var(x)=
    $$

  * 指数分布
    $$
    f(x)=\left\{\begin{aligned}
    \lambda e^{-\lambda x},x>0 \\
    0,x<=0
    \end{aligned}\right. \space,where\newline
    E(x)=\frac{1}{\lambda};\space Var(x)=\frac{1}{\lambda^2}
    $$
    
  * 正态分布 x ~ $N(\mu,\sigma^2)$
    $$
    f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}} \space,where\newline
    E(x)=\mu;\space Var(x)=\sigma^2
    $$
  
    * 转标准正态
  
      ![22343b2cbcb8af0d7df7ebc65c30cba](https://s2.loli.net/2024/05/14/VXymO3nTjt2QeGW.jpg)
  
* 随机变量函数概率分布 (已知X，求Y=h(X)的分布) --> 以Y=2X+1为例
  $$
  F(Y)=P(Y<=y)=P(2X+1<=y)=P(X<=\frac{y-1}{2})
  $$

  * 将事件Y<=y转化为已知事件$X<=\frac{y-1}{2}$的概率来求解



### 2维随机向量





## 高等数学

* [参考](https://zhuanlan.zhihu.com/p/707336465)



### 极限 (微分与积分的基础)

* 定义
  $$
  1. \lim_{x \to x_0} f(x) = A \leftrightarrow \forall \epsilon>0,\exist \delta , s.t. \space if \space 0<|x-x_0|<\delta, then |f(x)-A|<\epsilon \\
  2. 若f(x)当x\to x_0极限为零, f(x)为x\to x_0时的无穷小 \\
  3. 若f(x)当x\to x_0绝对值大于任何事先指定的正整数M, f(x)为x\to x_0时的无穷大
  $$

* (由定义推导出的)常用定理
  $$
  1. f(x)在x \to x_0是无穷大, \frac{1}{f(x)}在x \to x_0是无穷小 \\
  2. 有限无穷小的和是无穷小 \\
  3. 有界函数与无穷小的积是无穷小 \\
  4. 极限存在则唯一 \\
  \\
  夹逼定理: 在U(x_0)上,恒存在g(x)\geq f(x)\geq k(x), 若\lim_{x\rightarrow x_0}g(x)=\lim_{x\rightarrow x_0}k(x)=c,则\lim_{x\rightarrow x_0}f(x)=c\\
  \\
  极限的四则运算: limf(x)=a,limg(x)=b,则 \\
  lim[f(x)\pm g(x)] = limf(x)\pm limg(x); \\
  lim[f(x)*g(x)] = limf(x)*limg(x); \\
  lim[cf(x)] = climf(x)
  $$
  
* 一些常用极限
  $$
  \lim_{x\rightarrow 0}\frac{\sin x}{x}= 1\\
  \lim_{x\rightarrow 0}\frac{\ln (1+x)}{x}= 1\\
  \lim_{x\rightarrow 0}\frac{e^x - 1}{x}= 1\\
  \lim_{x\rightarrow 0}\frac{(1 + x)^\mu - 1}{x}= \mu\\
  \lim_{x\rightarrow 0}(1 + \frac{1}{x})^x=\lim_{y\rightarrow +\infty }(1 + y)^{\frac{1}{y}}= e = 2.718... \\
  \lim_{x\rightarrow 0}\frac{\log_a(1 + x)}{x}= \frac{1}{\ln a}\\
  \lim_{x\rightarrow 0}\frac{a^x - 1}{x}= \ln a\\
  $$



### 导数 (重点，需要刷题)

* 定义 
  $$
  f'(x_0) = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x} \\
  f'(x_0) = y'|_{x=x_0} = \frac{dy}{dx}, \quad \text{where } dy = f'(x)\, dx,\ \text{and } dy \approx \Delta y\ \text{only when } dx \to 0
  $$
  
* 定理(通过定义推导)
  $$
  [u(x)\pm v(x)]' = u'(x) \pm v'(x) \\
  [u(x)\pm v(x)]' = u'(x)v(x)+v'(x)u(x) \\
  [\frac{u(x)}{v(x)}]' = \frac{u'(x)v(x)-u(x)v'(x)}{v^2(x)} \\
  (f^{-1}(x)表示f(x)的反函数),\space [f^{-1}(x)'] = \frac{1}{f'(y)} \\
  (令u = g(x)), \space f[g(x)]' = f'(u)g'(x)
  $$
  
* 反函数求导应用arcsinx
  $$
  y=\sin x,\space 求\space arcsin'(x) \\
  \sin'(x) = \cos(x)=\frac{dy}{dx} = \frac{1}{\frac{dx}{dy}},\frac{dx}{dy}=\frac{1}{\cos x} \\ 
  \because \frac{dx}{dy}自变量为y\space \therefore顾需用y表示\cos x \\
  y=\sin x, 1 = \sin^2 x + \cos^2 x\rightarrow \cos^2x = 1-y^2 \\
  (\arctan x)'=\frac{1}{1+x^2}
  $$
  
* 基本求导公式
  $$
  (c)'=0 \newline\newline
  (\sin(x))'=\cos(x) \newline\newline
  (\cos(x))'=-\sin(x) \newline\newline
  (\tan(x))'=\sec^2x \newline\newline
  (x^a)'=ax^{a-1} \newline\newline
  (\arctan x)'=\frac{1}{1+x^2} \newline\newline
  (\arcsin x)'=\frac{1}{\sqrt{1-x^2}} \newline\newline
  (\arccos x)'=-\frac{1}{\sqrt{1-x^2}} \newline\newline
  (a^x)'=a^x\ln a\space,(e^x)'=e^x \newline\newline
  (\log_ax)'=\frac{1}{x\ln a}\space,(\ln x)'=\frac{1}{x} \newline
  $$
  
* 导数运算化简
  $$
  [f(x)\pm g(x)]'=f'(x)\pm g'(x) \newline
  [f(x)g(x)]'=f'(x)g(x)+f(x)g'(x) \newline
  [\frac{f(x)}{g(x)}]'=\frac{f'(x)g(x)-f(x)g'(x)}{g^2(x)} \newline
  $$

* 复合函数求导
  $$
  \frac{dy}{dx} = \frac{dy}{du}.\frac{du}{dx} \newline\newline
  [f(g(x))]'=f'(g(x))g'(x) \longrightarrow先将f(g(x))对g(x)求导, 再单独把g(x)对x求导
  $$

* 隐函数求导 --> 将等式整理成$y'=f(x)$, 原式中的$y$求导会得到$y'$, 对$xy$求导会当作复合函数处理
  $$
  ex: 求e^y+xy-e=0函数的导数 \newline
  y'e^y+y+xy'=0\longrightarrow y'=\frac{y}{x+e^y}
  $$
  
* 参数方程
  $$
  若参数方程\left\{\begin{aligned}
  x=\lambda(t) \\
  y=\epsilon(t)
  \end{aligned}\right. \\
  一阶导\frac{dy}{dx}=\frac{\lambda'(t)}{\epsilon'(t)} \newline
  二阶导\frac{d^2y}{dx^2}=\frac{d(\frac{dy}{dx})}{dx}=\frac{\frac{d(\frac{dy}{dx})}{dt}}{\frac{dx}{dt}}=\frac{\lambda'(t)}{\epsilon'(t)}
  $$
  <img src="https://cdn.jsdelivr.net/gh/laotianye01/img_bed@image/note_image/image-20250710172141572.png" alt="image-20250710172141572" style="zoom:50%;" />



### 定理与证明

* 微分中值定理

  <img src="https://cdn.jsdelivr.net/gh/laotianye01/img_bed@image/note_image/image-20250710172229045.png" alt="image-20250710172229045" style="zoom:50%;" />

* 拉格朗日中值定理

  <img src="https://cdn.jsdelivr.net/gh/laotianye01/img_bed@image/note_image/image-20250710172306134.png" alt="image-20250710172306134" style="zoom:50%;" />

* 洛必达定理
  $$
  \lim_{x\rightarrow0}\frac{f(x)}{g(x)}=\lim_{x\rightarrow0}\frac{f'(x)}{g'(x)}
  $$
  <img src="https://cdn.jsdelivr.net/gh/laotianye01/img_bed@image/note_image/image-20250710172332329.png" alt="image-20250710172332329" style="zoom:50%;" />

### 积分 (重点，需要刷题)

* 不定积分
  $$
  定义:\int f(x)dx=F(x)+C \newline\newline\newline
  \int \frac{1}{x}dx=ln(x)+C \newline\newline
  \int a^xdx=\frac{a^x}{lna}+C \newline\newline
  \int x^kdx=\frac{x^{k+1}}{k+1}+C \newline\newline
  \int \frac{1}{\sqrt{a^2-x^2}}dx=\arcsin (\frac{x}{a})+C \newline
  \int \frac{1}{a^2+x^2}dx=\frac{1}{a}\arctan (\frac{x}{a})+C \newline
  \int \frac{x}{\sqrt{a^2+x^2}}dx=\sqrt{a^2+x^2}+C \newline\newline
  \int \sin (x)dx=-\cos(X)|+C \newline
  \int \cos (x)dx=\sin(X)|+C \newline
  \int \tan (x)dx=-\ln|\cos(X)|+C \newline
  $$
  
* 一些积分方法

  * 换元法: 使用$f(u)=x$替换掉原有的x

    ![image-20240514134431475](https://s2.loli.net/2024/05/14/wvb7EkL4gxVzh9M.png)

  * 分布积分法 (在v更好积分的时候才使用 --> **比如在第一步时积分号内同时出现了x与任意三角函数**)
    $$
    \int udv = uv - \int vdu
    $$

* 定积分 (与不定积分对应)

  * 牛顿莱布尼茨公式
    $$
    \int_a^bf(X)dx=F(b)-F(a)
    $$

  * 换元法
    $$
    \int_a^bf(X)dx=\int_\alpha^\beta f(\omega(t))d\omega(t) \space,where\newline
    \alpha,\beta由原先x区域取值映射而得
    $$
    <img src="https://cdn.jsdelivr.net/gh/laotianye01/img_bed@image/note_image/image-20250710172052461.png" alt="image-20250710172052461" style="zoom:50%;" />

  * 部分积分法 (与不定积分对应)
    $$
    \int_a^b u(x)dv(x) = [u(x)v(x)]_a^b - \int_a^b v(x)du(x)
    $$
    <img src="https://cdn.jsdelivr.net/gh/laotianye01/img_bed@image/note_image/image-20250710172019170.png" alt="image-20250710172019170" style="zoom:50%;" />



### 微分方程 (忽略)

* 一阶线性

  



## 多元函数与空间几何



### 向量

* 数量积 -- 结果为数值
  $$
  1. 定义： \vec{a} \cdot \vec{b} = |\vec{a}| |\vec{b}| \cos \theta \\
  2. 计算： \vec{a} \cdot \vec{b} = a_x b_x + a_y b_y + a_z b_z \\
  3. 几何意义： \vec{a} \cdot \vec{b} = 0 \Leftrightarrow \vec{a} \perp \vec{b}
  $$

* 向量积 -- 结果为向量(叉乘)
  $$
  1. 定义： |\vec{c}| = |\vec{a}| |\vec{b}| \sin \theta \\
  2. 计算： \vec{a} \times \vec{b} = \begin{vmatrix} \vec{i} & \vec{j} & \vec{k} \\ a_x & a_y & a_z \\ b_x & b_y & b_z \end{vmatrix} \\
  3. 几何意义： \vec{a} \times \vec{b} = \vec{0} \Leftrightarrow \vec{a} \parallel \vec{b}
  $$
  
* 混合积 -- 结果为数值
  $$
  1. 定义： [\vec{a} \vec{b} \vec{c}] = (\vec{a} \times \vec{b}) \cdot \vec{c} \\
  2. 计算： [\vec{a} \vec{b} \vec{c}] = \begin{vmatrix} a_x & a_y & a_z \\ b_x & b_y & b_z \\ c_x & c_y & c_z \end{vmatrix} \\
  $$

* 空间面的方程

  $$
  平面的点法式方程：
  A(x - x_0) + B(y - y_0) + C(z - z_0) = 0 \\
  
  平面的一般方程：
  Ax + By + Cz + D = 0 \\
  
  平面的截距式方程：
   \frac{x}{a} + \frac{y}{b} + \frac{z}{c} = 1  \\
  $$
  
* 空间线方程
  $$
  直线的点向式方程：
  \frac{x - x_0}{m} = \frac{y - y_0}{n} = \frac{z - z_0}{p} \\
  
  直线的参数方程：
   \begin{cases} 
  x = x_0 + mt \\
  y = y_0 + nt \\
  z = z_0 + pt 
  \end{cases} \\
  $$

* 夹角
  $$
  线面角：
   \sin \theta = \frac{|Am + Bn + Cp|}{\sqrt{A^2 + B^2 + C^2} \sqrt{m^2 + n^2 + p^2}} \\
   面面角：
   \cos \theta = \frac{|A_1A_2 + B_1B_2 + C_1C_2|}{\sqrt{A_1^2 + B_1^2 + C_1^2} \sqrt{A_2^2 + B_2^2 + C_2^2}}
  $$

* 向量值函数(1~n维)定义
  $$
  \vec{f}(x) = 
  \begin{bmatrix}
  f_1(x) \\
  f_2(x) \\
  \vdots \\
  f_m(x)
  \end{bmatrix}
  =
  \begin{bmatrix}
  f_1(x_1, x_2, \cdots, x_n) \\
  f_2(x_1, x_2, \cdots, x_n) \\
  \vdots \\
  f_m(x_1, x_2, \cdots, x_n)
  \end{bmatrix}
  $$

* 向量值函数极限与微分定义
  $$
  极限: \lim_{x \to x_0}\vec{f}(x) = \vec{a} \space \leftrightarrow \space\vec{f}(x) - \vec{a}\| = \sqrt{\sum_{i=1}^{m}(f_i(x) - a_i)^2} < \varepsilon \\
  微分:设 \vec{f} = (f_1(x), f_2(x), \cdots, f_m(x))^T \\
  \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x} \text{ 存在} \Leftrightarrow \lim_{\Delta x \to 0} \frac{f_i(x_0 + \Delta x) - f_i(x_0)}{\Delta x} \text{ 存在} \\
  \Leftrightarrow \\f_1(x), f_2(x), \cdots, f_m(x) 均在 x 可导
  $$

* 方向导数定义(可想象为一个三维空间中的曲面，自变量x0,x1在x,y平面内内沿方向l移动了一个极小量)
  $$
  x_0\in R^2(x_0为二维平面中一点),有二元函数f:U(x)\subset R^2 \to R 与向量 \vec{l},其单位向量为 \vec{e_i} \\ 过x_0直线为x=x_0+t\vec{e_i},t\in R \\ \vec{e_i}方向的变化率为\frac{\partial f}{\partial \vec{l}} = \lim_{t \to 0} \frac{f(x_0 + t\vec{e_i}) - f(x_0)}{t}
  $$

* 梯度定义
  $$
  定义: 对于函数u=f(x_1,...,x_n) \\
  梯度grad: gradf(x_0)=\Lambda f(x_0) = (\frac{\partial f(x_0)}{\partial x_1},...,\frac{\partial f(x_0)}{\partial x_n})\\
  性质:对应函数u=f(x_1,...,x_n)在梯度方向上的变化率最快
  $$

* 方向导数有关定理
  $$
  1. 方向:对应方向导数取最大值的方向 \\
  2. 模:方向导数的最大值 \\
  3. x_0处\vec{e_i}方向上的方向导数值为\vec{e_i}\cdot gradf(x_0) \leftrightarrow (\vec{e_i}点乘gradf(x_0))
  $$
  

### 多元函数 

* 多元函数导数 -- 分别对于每个变量求解偏导数(将其余变量当作常量处理) -- 曲面对应点x/y方向切线的斜率
  $$
  f(xy) = x + xy + y \newline
  f_x(xy) = 1 + y \newline
  f_y(xy) = 1 + x
  $$

* 全微分定义
  $$
  f在(x_0,y_0)可微 \\ \leftrightarrow \\ z=f(x,y)在(x_0,y_0)某临域U(x_0,y_0)内有定义,当(x_0+\Delta x,y_0+\Delta y) \in U(x_0,y_0) \\ 且\Delta z = f(x_0+\Delta x,y_0+\Delta y) - f(x_0,y_0)可表示为\Delta z = A\Delta x+B\Delta y + o(p),p=\Delta x^2+y_0+\Delta y^2
  $$
  
* 全微分计算
  $$
  du=\frac{\partial u}{\partial x} + \frac{\partial u}{\partial y} + \frac{\partial u}{\partial z}
  $$
  
* 复合求导 (与一元相同)
  $$
  z = e^{xy}sin(x + y) \\ 
  z'_x = ye^{xy}sin(x + y) + e^{xy}cos(x + y) = e^{xy}(ysin(x+y)+cos(x+y))
  $$

* 多元函数极值 -- 满足A点处$x_0, y_0邻域内任意f(x，y) > or < f(x_0,y_0)$ -- 保证当前点不是类似如鞍点那样的形状 -- 至少保证两个偏导数是单调的(以下的证明较为复杂，此处不进行展开)
  $$
  f_x(x_0,y_0) = 0\space and \space f_y(x_0,y_0)=0 \newline
  
  D = \begin{bmatrix}
  f_{xx}(x_0,y_0) & f_{xy}(x_0,y_0) \\
  f_{xy}(x_0,y_0) & f_{yy}(x_0,y_0)
  \end{bmatrix} \newline
  极大值: D > 0 \space and \space f_{xx}(x_0,y_0) < 0 \newline
  极小值: D > 0 \space and \space f_{xx}(x_0,y_0) > 0
  $$

* 多元函数条件极值 -- 拉格朗日函数(当无法直接消元时)

* [原理参考](https://www.cnblogs.com/mo-wang/p/4775548.html)
  $$
  \left\{
  \begin{array}{l}
  z = f(x,y) \\
  0 = \epsilon(x,y) \rightarrow 约束条件 
  \end{array}
  \right.
  $$

  1. 引入拉格朗日函数
     $$
     L(x,y) = f(x,y) + \lambda\epsilon(x,y)
     $$

  2. 联立方程组
     $$
     \left\{
     \begin{array}{l}
     L'_x = f'_x(x,y) + \lambda\epsilon'_X(x,y) \\
     L'_y = f'_y(x,y) + \lambda\epsilon'_y(x,y) \\
     L'\epsilon = \epsilon(x,y) = 0
     \end{array}
     \right.
     $$
     



### 重积分

* 定义: 上下限可为函数，在对某一个变量积分的时候另一个变量为常数

  ![image-20240514112449553](https://s2.loli.net/2024/05/14/vfkLRgVMPsTuSji.png)

  ![image-20240514112503283](https://s2.loli.net/2024/05/14/ZlpkBSiGET13UHC.png)



### 级数 (忽略)

