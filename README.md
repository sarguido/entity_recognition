# Named Entity Recognition

This project contains two parts: the prediction API endpoint, and the web app with the project documentation.

### API

To launch the api, please make sure you have Docker installed.

Open your terminal and type 

`docker pull sarahguido/prediction_endpoint`

to pull down the image. It'll take a bit of time, but once that finishes, type the following:

`docker run -p 8000:8000 sarahguido/prediction_endpoint`

This will launch the API. For instructions on using the API, see the web app documentation for more details.

### Web App

Make sure you have `virtualenv` installed, or some other mechanism for activating a virtual environment, unless you're fine with installing everything in your local environment!

In another terminal tab, `cd` into the top-level directory of the project, `entity_recognition`, and create a virtual environment using Python 3 (for example, `python3 -m venv entity_project` if Python 2 is your default). 

Activate the virtual environment (for example, with `source entity_project/bin/activate`), and then install the project requirements using `pip install -r requirements.txt`.

Once that finishes, run `python main.py` and navigate in the browser to the localhost:port combo that appears as output. Mine is `http://0.0.0.0:5000`, for example.

You technically don't need everything in `requirements.txt` to run the web app, but I left everything in there in case you'd like to run the various `.py` files.