FROM pm4py/pm4py-core:latest@sha256:7cfd94ae6076b1e9e16a4decb6db412ab4acc900d6c20cff808b0cb783105624

WORKDIR /usr/src/app

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"
ENV DATA_PATH "/usr/data/"
ENV DATA_FILENAME "input"
ENV RANDOM_SAMPLE_SIZE "10000"
ENV TIMEOUT "10"
ENV NOISE_THRESHOLD "0.9"
ENV DEBUG "False"
ENV MODE "file"
ENV INFIX_TYPE "infix"

COPY . .

CMD [ "python", "./infix_alignments/experiments/infix_alignments/experiments.py" ]