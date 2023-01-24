import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import copy
import numpy as np
import tensorflow as tf
#from keras.models import load_model


import image_preprocesing as ip_utils
import solver


model = tf.keras.models.load_model("modelo.h5")


def predict_digits(digits, threshold=0.5):
    """
    Predicts digits using trained model. If predicted value has
    value below threshold, zero is predicted instead of the actual
    prediction. This handles the cases of noise in image detected
    as digits.
    """
    digits = digits / 255.
    if threshold == 0:
        return model.predict(digits.reshape(-1, 28, 28, 1))

    pred = model.predict(digits.reshape(-1, 28, 28, 1))
    predicted = []
    for p in pred:
        idx = np.argmax(p)
        predicted.append(idx if p[idx] > threshold else 0)
    return predicted


class Sudoku:
    def __init__(self, img_path=None, img=None, max_size=1000, predict_threshold=0.5, overlay=True):
        """
        Initializes the Sudoku grid from an image.
        Args:
            img_path (str): path to the image of sudoku
            img (numpy.ndarray): loaded image instead of img_path
            max_size (int): maximum allowed size of image
            predict_threshold (float): minimun threshold to predict digit
            overlay (bool): generate the output image
        """
        if img_path is None and img is None:
            raise ValueError("img_path and img both cannot be None.")
        if img_path is not None and img is not None:
            raise ValueError("Only one of img_path and img can be used.")
        if img_path:
            if not os.path.exists(img_path):
                raise FileNotFoundError(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        self.status = 'processing'
        self.original = ip_utils.scale_down(img, max_size)
        self.solved_image = None
        self.grayscale = cv2.cvtColor(self.original.copy(), cv2.COLOR_BGR2GRAY)

        # crop and warp to get the sudoku grid
        self.cropped, self.crop_matrix, self.crop_corners = ip_utils.crop_grid(self.grayscale)
        self.cropped_color = None

        # divide the grid into cells
        self.grid_cells = ip_utils.divide_into_cells(self.cropped)

        # extract the digits from the cells
        self.digits = ip_utils.get_digits(self.cropped, self.grid_cells, size=28)

        # convert the images of digits to digits using digit recognition model
        self.board = [[0] * 9 for _ in range(9)]
        self.solved_board = None
        pred_indices = []
        pred_digits = []
        for i, digit in enumerate(self.digits):
            # if digit is present in cell
            if cv2.countNonZero(digit) > 0:
                pred_digits.append(digit)
                pred_indices.append((i // 9, i % 9))
        predicted = predict_digits(np.array(pred_digits), threshold=predict_threshold)
        for (i, j), d in zip(pred_indices, predicted):
            self.board[i][j] = d

        # check if recognized sudoku puzzle is valid
        if not solver.is_valid(self.board):
            self.status = 'Unable to detect a valid Sudoku puzzle.'
            return

        self.solved_board = copy.deepcopy(self.board)

        if solver.solve(self.solved_board, validate=False):
            if overlay:
                self.solved_image = self.digits_overlay()
            self.status = 'solved'
        else:
            self.status = 'Either the sudoku is invalid or image is blurry.'

    def digits_overlay(self, color=(0, 0, 0), thickness=3):
        """Draws the digits on the orignal image."""
        if self.solved_image is not None:
            return self.solved_image

        if self.cropped_color is None:
            self.cropped_color, matrix, corners = ip_utils.crop_and_warp(self.original, self.crop_corners)
        img = self.cropped_color.copy()

        # font scale
        scale = int((self.grid_cells[0][1][0] - self.grid_cells[0][0][0]) * 0.06)
        for i, cell in enumerate(self.grid_cells):
            # if original cell was empty
            if self.board[i // 9][i % 9] == 0:
                digit = str(self.solved_board[i // 9][i % 9])
                fh, fw = cv2.getTextSize(digit, cv2.FONT_HERSHEY_PLAIN, scale, thickness)[0]
                h_pad = (cell[1][0] - cell[0][0] - fw) // 2
                v_pad = (cell[1][1] - cell[0][1] - fh) // 2
                img = cv2.putText(img, digit, (int(cell[0][0]) + h_pad, int(cell[1][1]) - v_pad),
                                  cv2.FONT_HERSHEY_PLAIN, fontScale=scale, color=color, thickness=thickness)

        # perform inverse crop and warp transformation to put image back onto original image
        height, width = self.original.shape[:2]
        img = cv2.warpPerspective(img, self.crop_matrix, (width, height), flags=cv2.WARP_INVERSE_MAP,
                                  dst=self.original.copy(), borderMode=cv2.BORDER_TRANSPARENT)

        return img

    def save_solved_image(self, path):
        cv2.imwrite(path, self.solved_image)

    def get_status(self):
        return self.status

    def get_original_image(self):
        return self.original

    def get_cropped_image(self):
        return self.cropped

    def get_extracted_digits_image(self):
        return ip_utils.display_digits(self.digits)

    def get_solved_image(self):
        return self.solved_image

    def get_solved_board(self):
        return self.solved_board

    def get_board(self):
        return self.board


#if __name__ == '__main__':
    # demo
 #   s = Sudoku('images/sudoku1.jpeg')
  #  cv2.imshow('Original', s.get_original_image())
   # cv2.imshow('Solved', s.get_solved_image())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()