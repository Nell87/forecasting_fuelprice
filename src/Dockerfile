FROM python:3

ADD requirements.txt . ./

RUN pip install -r requirements.txt

ADD collect_data.py . ./

CMD [ "python", "./collect_data.py" ]