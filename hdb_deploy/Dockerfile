FROM python:3.7

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

# exposing default port for streamlit
EXPOSE 8501

# making directory of app
WORKDIR /app

# copy over requirements
COPY requirements.txt ./requirements.txt

# install pip then packages
RUN pip install -r requirements.txt

# copying all files over
COPY . .

# cmd to launch app when container is run
CMD streamlit run app.py