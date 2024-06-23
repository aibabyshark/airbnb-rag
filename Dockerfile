# Use an official Python runtime as a parent image
FROM python:3.9

# Create a non-root user and switch to that user
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Update pip
RUN pip install --upgrade pip

# Copy the requirements file and install dependencies
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN pip install -r requirements.txt

# Create necessary directories with correct permissions
RUN mkdir -p $HOME/app/data/vectorstore && chown -R user:user $HOME/app/data
COPY . .

# Specify the command to run the application
CMD ["chainlit", "run", "app.py", "--port", "7860"]
