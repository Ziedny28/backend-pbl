# How to Use

## Setup

instruction below is used for windows

- create venv 
  ```
  python -m venv venv
  ```
- activate venv
  ```
    venv\Scripts\activate
  ```
- install dependencies for flask
  ```
    pip install -r requirements.txt
  ```
- install pytesseract
  ```
!sudo apt install tesseract-ocr
!pip install pytesseract
!sudo apt-get install tesseract-ocr-ind
  ```

## Run the Local Server

- activate venv(if it hasnt active)
  ```
    venv\Scripts\activate
  ```
- go to ocr directory
  ```
    cd ocr
  ```
- start server
 ```
 python -m flask run --port=5006 --host=0.0.0.0
 ```

