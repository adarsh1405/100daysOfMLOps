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
Load model from the registry with custom pre-processing

```python
# TODO1
        X_scaled = (X - self.mean) / self.std
        return self.model.predict(X_scaled)

# TODO2
inner_model = mlflow.pyfunc.load_model(MODEL_URI)

# TODO3
preds = predictor.predict(None, inputs.values) 
inputs["prediction"] = np.asarray(preds).reshape(-1)
inputs.to_csv(OUTPUT_CSV, index=False)

```

### Day 28
Debug mlflow project & re-run it

``` shell
root@controlplane ~/code/trainer via 🐍 v3.12.3 ➜  cat /tmp/mlflow-run-initial.log
2026/07/12 07:11:40 INFO mlflow.projects: 'trainer' does not exist. Creating a new experiment
2026/07/12 07:11:40 INFO mlflow.projects.utils: === Created directory /tmp/tmpkgj8wag3 for downloading remote URIs passed to arguments of type 'path' ===
2026/07/12 07:11:40 INFO mlflow.projects.backend.local: === Running command 'python train.py --n_est 100' in run with ID '9a46ca1bca7841b98c4c047cedf71201' === 
usage: train.py [-h] [--n_estimators N_ESTIMATORS] [--max_depth MAX_DEPTH]
                [--test_size TEST_SIZE] [--random_seed RANDOM_SEED]
train.py: error: unrecognized arguments: --n_est 100
2026/07/12 07:11:42 ERROR mlflow.cli: === Run (ID '9a46ca1bca7841b98c4c047cedf71201') failed ===
🏃 View run rebellious-snake-745 at: http://localhost:5000/#/experiments/1/runs/9a46ca1bca7841b98c4c047cedf71201
🧪 View experiment at: http://localhost:5000/#/experiments/1

```

`Edit the MLproject`
```yaml
command: >
      python train.py
      --n_estimators {n_estimators}
      --max_depth {max_depth}
      --test_size {test_size}
      --random_seed {random_seed}
```

```shell
# Run the foloowing command
mlflow run . -e train --env-manager local
mlflow run . -e train -P n_estimators=200 -P max_depth=10 --env-manager local
```


### Day 29

Fix MLflow's Remote Artifact-Store Wiring (PostgreSQL + SeaweedFS)


```shell
# Debug commands for the postgresQL
docker exec mlflow-db psql -U mlflow -d mlflow --version
docker exec -it mlflow-db psql -U mlflow -d mlflow 
```


```shell
# Added/Edited these lines in startup script
export MLFLOW_S3_ENDPOINT_URL="http://localhost:8333"

# Added --serve-artifacts 
exec mlflow server \
  --backend-store-uri postgresql://mlflow:mlflow123@localhost:5432/mlflow \
  --artifacts-destination s3://mlflow-artifacts \
  --serve-artifacts \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts '*' --cors-allowed-origins '*'

```
