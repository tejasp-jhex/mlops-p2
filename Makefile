mlflow-ui:
	py -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

