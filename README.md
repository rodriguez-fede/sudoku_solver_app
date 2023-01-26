# sudoku_solver_app

This program serves as a way to calculate the solution to any 9x9 sudoku puzzle. It identifies the puzzle with OpenCV, runs against a convolucional neural network to predict the digits, and runs an efficient sudoku solver to determine the answer using a backtracking algorithm. Finaly it displays the answer.
---


---
pip install -r requierements.txt

---
python app.py
---
---




# sudoku_solver_app
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/) 

Web app Smart Sudoku Solver that tries to extract a sudoku puzzle from a photo and solve it.

## Table Of Contents:

[Installation](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/README.md#installation)

[Usage](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/README.md#usage)

[Working](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/README.md#working)

  * [Image Preprocessing](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/README.md#image-preprocessing)
  * [Recognition](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/README.md#recognition)

[ToDo](https://github.com/neeru1207/AI_Sudoku/blob/master/README.md#todo)

[Contributing](https://github.com/neeru1207/AI_Sudoku/blob/master/README.md#contributing)

## Installation

1. git clone the project in a folder that you con open with VScode (I recomend)
2.  I recomend using a conda virtual enviroment, and in there install the requirements and run the app.
    ```bash
    pip install -r requirements.txt
    ```
3. And in the terminal of the venv run the app
    ```bash
    python app.py
    ```
   
## Usage
* Runing python app.py will open a local host that there you can upload a sudoku image. That image will be proccess and after the digit recognition and the backtracking algorith the answer will be display in the same image thta the user upload.

* The web Homepage that opens up as soon as you run the application.
    ![](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/images/1.png)  

* Upload a sudoku image to get the solution
    ![](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/images/2.png)    

## Working

### Image Preprocessing

* **Gaussian Blurring**  and **Adaptive Gaussian Thresholding**Blurring using a Gaussian function. This is to reduce noise and detail. Adaptive thresholding with a Gaussian Function to account for different illuminations in different parts of the image.

    ![](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/images/3.png) 


* **Dilation** with a plus shaped 3X3 Kernel to fill out any cracks in the board lines and thicken the board lines.

* **Flood Filling** Since the board will probably be the largest blob a.k.a connected component with the largest area, floodfilling from different seed points and finding all connected components followed by finding the largest floodfilled area region will give the board. 


* The **largest blob** a.k.a the board is found after the previous step. Let's call this the outerbox


* **Eroding** the grid a bit to undo the effects of the dilation on the outerbox that we did earlier.


* **Hough Line Transform** to find all the lines in the detected outerbox.

    

* **Merging** related lines. The lines found by the Hough Transform that are close to each other are fused together.

* **Finding the Extreme lines** . We find the border lines by choosing the nearest line from the top with slope almost 0 as the upper edge, the nearest line from the bottom with slope almost 0 as the lower edge, the nearest line from the left with slope almost infinity as the left edge and the nearest line from the right with slope almost infinity as the right edge.

    ![](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/images/5.png) 

* **Finding the four intersection points**. The four intersection points of these lines are found and plotted along with the lines.



* **Warping perspective**. We find the perspective matrix using the end points, correct the perspective and crop the board out of the original image.

    ![](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/images/4.png) 

* **Thresholding and Inverting the grid**. The cropped image from the previous step is adaptive thresholded and inverted.


* **Slicing** the grid into 81 slices to get images of each cell of the Sudoku board.

    ![](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/images/6.png) 

* **Blackfilling and centering the number**. Any white patches other than the number are removed by floodfilling with black from the outer layer points as seeds. Then the approximate bounding box of the number is found and centered in the image.

    ![](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/images/7.png) 

    ![](https://github.com/rodriguez-fede/sudoku_solver_app/blob/main/images/8.png) 

### Recognition

#### Convolutional Neural Network

Read about CNNs [here](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
* **Layers** A Convolution Layer, each of the inner layer uses *ReLu* while the output layer uses *softmax*.
* **Compilation** "rmsprop" optimizer and *categorical cross entropy* loss.
* **Training** The model is trained on the **MNIST** handwritten digits dataset which has around 70,000 28X28 images.
* **Accuracy** Around 99 percent accuracy on the test set.

    
## ToDo

* Improve Accuracy.
* Resolve Bugs/Issues if any found.
* Optimize Code to make it faster.

## Contributing

Contributions are welcome :smile:

### Pull requests

Just a few guidelines:
* Write clean code with appropriate comments and add suitable error handling.
* Test the application and make sure no bugs/ issues come up.
* Open a pull request and I will be happy to acknowledge your contribution after some checking from my side.

### Issues

If you find any bugs/issues, raise an issue.


