FROM python:3.10.9

WORKDIR /app

# copy model and app.py
COPY ./model model
COPY ./app.py app.py
COPY ./templates templates
COPY ./static static

# pip install dependency
COPY ./requirements.txt /var/www/requirements.txt
RUN pip install -r /var/www/requirements.txt

# run flask server on default host
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]