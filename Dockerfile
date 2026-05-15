FROM python:3.11-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code and pre-built outputs
COPY api/        ./api/
COPY outputs/    ./outputs/
COPY streamlit_app/ ./streamlit_app/

# expose FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# default command starts the API
# override CMD to run Streamlit instead
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]