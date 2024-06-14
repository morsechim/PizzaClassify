# Pizza Classification 🍕

Welcome to the **Pizza Classification** project! This project utilizes a YOLOv8 model from [Ultralytics](https://ultralytics.com/) to classify different types of pizzas from uploaded images. The application is built using [Streamlit](https://streamlit.io/), making it easy to interact with the model via a web interface.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with the project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/morsechim/PizzaClassify
    cd PizzaClassify
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the YOLO model:**

    Place the `pizza_classify.pt` model file in the `weights` directory. Ensure the directory structure looks like this:

    ```
    pizza-classification/
    ├── weights/
    │   └── pizza_classify.pt
    ├── app.py
    └── requirements.txt
    ```

## Usage

To run the application, use the following command:

```bash
streamlit run app.py
