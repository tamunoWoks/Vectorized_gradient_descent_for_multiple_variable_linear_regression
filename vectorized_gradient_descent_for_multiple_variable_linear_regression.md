## Multiple Variable Linear Regression
In this project, we will implement the data structures and routines to support linear regression for multiple features.
#### Goals
- Implement our regression model routines to support multiple features
    - Implement data structures to support multiple features
    - Write prediction, cost and gradient routines to support multiple features
    - Utilize NumPy np.dot to vectorize their implementations for speed and simplicity
#### Tools
In this project, we will make use of:
- NumPy, a popular library for scientific computing
- Matplotlib, a popular library for plotting data
```python
import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays
```
#### Notation
Here is a summary of some of the notation we will encounter, updated for multiple features.
| **General Notation** | **Description** | **Python (If applicable)** |
|----------------------|-----------------|----------------------------|
| *a* | scalar, non bold | |
| **a** | vector, bold | |
| **A** | matrix, bold capital | |
| **Regression** | | |
| **X** | training example matrix | X_train |
| **y** | training example targets | y_train |
| **x**<sup>(i)</sup>, ***y***<sup>(i)</sup> | ***i***<sub>th</sub>Training Example | X[i], y[i] |
| m | number of training example | `m` |
| n | number of training example | `n` |
| **w** | parameter: weight, | `w` |
| *b* | parameter: bias | `b` |
| *𝑓*<sub>𝐰,𝑏</sub>(**𝐱**<sup>(𝑖)</sup>) | The result of the model evaluation at  𝐱<sup>(𝐢)</sup> parameterized by 𝐰,𝑏: *𝑓*<sub>𝐰,𝑏</sub>(𝐱<sup>(𝑖)</sup>)=𝐰⋅𝐱(<sup>(𝑖)</sup>)+𝑏 | `f_wb` |
#### Problem Statement
We will use the housing price prediction example. The training dataset contains three examples with four features (size, bedrooms, floors and, age) shown in the table below.  Note the size is in sqft rather than 1000 sqft.
| Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |   
| ----------------| ------------------- |----------------- |--------------|-------------- |  
| 2104            | 5                   | 1                | 45           | 460           |  
| 1416            | 3                   | 2                | 40           | 232           |  
| 852             | 2                   | 1                | 35           | 178           |  

Let us build a linear regression model using these values so we can then predict the price for other houses. For example, a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.  

Let's run the following code cell to create your `X_train` and `y_train` variables.
```python
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
```
#### Matrix X containing our examples
Similar to the table above, examples are stored in a NumPy matrix `X_train`. Each row of the matrix represents one example. When you have 𝑚 training examples ( 𝑚 is three in our example), and there are 𝑛 features (four in our example), 𝐗 is a matrix with dimensions (**𝑚**, **𝑛**) (m rows, n columns).  

notation:
- 𝐱<sup>(𝑖)</sup> is vector containing example i. 𝐱<sup>(𝑖)</sup> =(𝐱<sup>(𝑖)</sup><sub>0</sub>,𝐱<sup>(𝑖)</sup><sub>1</sub>,⋯,𝐱<sup>(𝑖)</sup><sub>𝑛−1</sub>).  
- 𝐱<sup>(𝑖)</sup><sub>j</sub> is element j in example i. The superscript in parenthesis indicates the example number while the subscript represents an element.  

Display the input data.
```python
# data is stored in numpy array/matrix
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)
```
**Output:**  
X Shape: (3, 4), X Type:<class 'numpy.ndarray'>)  
[[2104    5    1   45]  
 [1416    3    2   40]  
 [ 852    2    1   35]]  
y Shape: (3,), y Type:<class 'numpy.ndarray'>)  
[460 232 178]  
