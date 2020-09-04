FROM python:alpine3.7  
COPY . /app
WORKDIR /app
FROM python:3.8-alpine
RUN apk --update add gcc build-base freetype-dev libpng-dev openblas-dev
RUN pip install --no-cache-dir matplotlib pandas
RUN pip3 install numpy scipy pandas matplotlib
RUN pip3 install -r requirements.txt 
EXPOSE 5001 
ENTRYPOINT [ "python" ] 
CMD [ "python","main.py" ]