import os

from flask import Flask, request, render_template

from sudoku import Sudoku

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

ALLOWED_IMAGE_EXTENSIONS = {'jpeg', 'jpg', 'png', 'jpe', 'bmp', 'dib', 'jp2', 'pbm', 'pgm', 'tiff', 'tif'}
INPUT_IMAGE_PATH = 'static/input.jpg'
OUTPUT_IMAGE_PATH = 'static/output.jpg'


def allowed_image(name):
    """Checks if file has a valid extension."""
    return '.' in name and name.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    # delete old input and output images
    if os.path.exists(INPUT_IMAGE_PATH):
        os.remove(INPUT_IMAGE_PATH)
    if os.path.exists(OUTPUT_IMAGE_PATH):
        os.remove(OUTPUT_IMAGE_PATH)

    if request.method == 'POST':
        image = request.files['image']
        if image and allowed_image(image.filename):
            image.save(INPUT_IMAGE_PATH)
            sudoku = Sudoku(img_path=INPUT_IMAGE_PATH)
            status = sudoku.get_status()
            if status == 'solved':
                sudoku.save_solved_image(OUTPUT_IMAGE_PATH)
                return render_template('index.html', input_img=True, output_img=True)
            else:
                return render_template('index.html', input_img=True, status=status)
        else:
            return render_template('index.html', error=True)
    return render_template('index.html')

@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response



if __name__=='__main__':
    app.run(debug=True)

