### Task 20

```bash
### run the mlflow server
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts "*" \
    --cors-allowed-origins "*" \
    --backend-store-uri sqlite:////root/code/mlflow-backend/mlflow.db \
    --default-artifact-root /root/code/mlflow-artifacts/
```


### Task 21

```bash
#TODO_1
mlflow.log_params(params) # stores dictonary or list

#TODO_2
mlflow.log_metric("accuracy", accuracy)  ## to log the metrics 
mlflow.log_metric("f1_score", f1)

#TODO_3
mlflow.sklearn.log_model(sk_model=model) # Stores model information & that version artifacts

```


### Task 22

- Using of UI , create 2 experiment , add tags 


### Task 23

- Comapre the experiments & add tags by searching through the query in the UI


### Task 24

- `mlflow.sklearn.autolog()` // Set autolog for a specific package
- `mlflow.set_experiment("autolog-demo")` // Set the experiment


### Task 25
Register , version & manage model lifecycle

- Create a model registry
- Register your model from the UI
- Give alias
- Compare the model


### Task 26
Compare model runs & select the best

```python
# TODO1
signature = mlflow.models.infer_signature(X, preds)

# TODO2
mlflow.sklearn.log_model(
    model,
    name="model",
    input_example=X,      # Generates required signature
    signature=signature,  # Or provide explicit signature
    )
```

### Task 27

