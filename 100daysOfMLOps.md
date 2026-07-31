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


### Day 30
End-to-End MLflow: Register, Serve, and Monitor the Champion



Register the model from the UI , with given format & criteria

```shell
export MLFLOW_TRACKING_URI=http://localhost:5000

mlflow models serve -m models:/fraud-detector-v2/1 -p 5001 --env-manager=local
```


```curl localhost:5001/health -vvv```


**monitor.sh**

```shell
#!/usr/bin/env bash
set -u

URL="http://127.0.0.1:5001/health"

code=$(curl -s -o /dev/null -w '%{http_code}' "$URL")
if [ "$code" = "200" ]; then
  exit 0
else
  exit 1
fi
```



### Day 31
Fix a Broken Config-Driven Training Setup

Edit in **train_config.yaml**
```yaml
type: RandomForestClassifier
```
```yaml
data:
  target_column: is_fraud
```
```yaml
output:
  model_path: /root/code/fraud-detection/models/model.pkl
```


### Day 32
Make a Training Script Reproducible (Seed Discipline)


```python
# Added random_state in split & model calling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
```

### Day 33
Fix a Broken Evaluation Script and Metrics Report

update the `train.py`

```python
METRICS_JSON = os.path.join(REPORTS_DIR, "metrics.json")
```
```python
    metrics = {
        "accuracy": round(accuracy_score(y, preds), 6),
        "precision": round(precision_score(y, preds), 6),
        "recall": round(recall_score(y, preds), 6),
        "f1_score": round(f1_score(y, preds), 6),
        "auc_roc": round(roc_auc_score(y, preds), 6)
    }
```




### Day 34
Fix a Broken Cross-Validation Loop (Stratified + Aggregates)
```python
# TODO 1
fold = {
    "fold": fold_idx,
    "accuracy": round(accuracy_score(y_test, preds), 6),
    "f1": round(f1_score(y_test, preds), 6),
    "roc_auc": round(roc_auc_score(y_test, proba), 6),
}

# TODO 2
with mlflow.start_run(run_name=f"fold-{fold_idx}", nested=True):
    mlflow.log_param("fold", fold_idx)
    mlflow.log_metric("accuracy", fold["accuracy"])
    mlflow.log_metric("f1", fold["f1"])
    mlflow.log_metric("roc_auc", fold["roc_auc"])
```



### Day 35 (Important)
Fix a Broken Optuna Tuner with MLflow Logging

```python
# Fix 1 : Track the log for each trial
with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("f1_score", score)
```

```python
# Fix 2: Replace the minimize to maxmimze , as we need the value for the highest
study = optuna.create_study(direction="maximize", study_name=EXPERIMENT_NAME)
```




### Day 36
Fix a Multi-Model Bake-Off in the MLflow Compare View


```python
    # Replace ASC to DESC as we need the highest f1_score
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=10,
    )
```
```python
    winner = runs.iloc[0]
    report = {
        "run_id": winner["run_id"],
        "f1_score": float(winner["metrics.f1_score"]),
        # Added the model_type to the json
        "model_type": winner["tags.mlflow.runName"]
    }
```



### Day 37 (Important)
Fix a Four-Stage Training Pipeline's Inter-Stage Wiring

```python
# featurize.py
input_path = config["data"]["processed_path"]
```
```python
train.py
features_path = config["data"]["features_path"]
```

```python
# run_pipeline.py
# TODO 1
mlflow.log_param("model_type",config["model"]["type"])
mlflow.log_param("n_estimators",config["model"]["n_estimators"])
mlflow.log_param("max_depth",config["model"]["max_depth"])


# TODO 2
metrics = config["output"]["report_path"]
with open(metrics, "r") as file:
    data = json.load(file)
mlflow.log_metrics(data)

```



### Day 38:
Fix a Parallel-Training Bake-Off (n_jobs Backend)


```python
# train.py
N_JOBS_VALUES = [1, -1]
mlflow.log_param("n_jobs", n_jobs)


with mlflow.start_run(run_name="speedup-summary"):
        mlflow.log_metric("speedup",(times[1] / times[-1]))
```


### Day 39: (Good one) - you can run on CPU or in GPU based upon your needs
Make a PyTorch Trainer Device-Aware with Checkpointing

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = model.to(device)


mlflow.log_param("device", device)
xb = X_t.to(device)
yb = y_t.to(device)

```
```python
# TODO
if epoch % 10 == 0:
    torch.save(
    {'epoach': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': final_loss
    },
    os.path.join(CHECKPOINT_DIR,f"ckpt_epoch_{epoch}.pt")
    )

```


### Day 40: 
Fix and Complete a Five-Stage Training Capstone



```shell
  # Makefile
  # Re-arrange the file 
  python3 src/validate_data.py
  python3 src/tune.py
  python3 src/select_model.py
  python3 src/register.py
  python3 src/report.py
```

```python
# select_model.py
# Replaced accuracy to f1_score 
runs = mlflow.search_runs(
      experiment_ids=[exp.experiment_id],
      order_by=["metrics.f1_score DESC"],
      max_results=200,
  )

score = float(best["metrics.f1_score"])

```

```python
# register.py
# TODO
client.set_registered_model_alias(REGISTERED_MODEL_NAME, RELEASE_ALIAS, version.version)
```

```python
# report.py
report = {
    "best_model": selection["model_type"],
    "best_params": best_params,
    "metrics": best_metrics,
    "total_trials": total_trials,
    "validation_status": validation["status"]
}
```


### Day 41
Scaffold a Feast Feature Repository and Build a Training Set

```shell
feast init feature_repo
cd feature_repo/feature_repo
feast apply
```

```shell
feast ui & // to run the Feast UI
```


```python
# replace training_df line with this 
training_df = store.get_historical_features(entity_df=entity_df, features=["driver_hourly_stats:conv_rate", "driver_hourly_stats:acc_rate", "driver_hourly_stats:avg_daily_trips"]).to_df()

```

``` shell
python  build_training_set.py 
```

### Day 42
Define a Feast Feature View (Entity + Field Schema)

```python
join_keys=["customer_id"],
```
```python 
Field(name="hour", dtype=Int64),
Field(name="num_tx_past_day", dtype=Int64),
```




### Day 43


```shell
# ./materialize.sh
END_DATE="2026-01-01T00:00:00"
```


```python
# fetch_features.py
result = store.get_online_features(
    features=[
      "customer_transaction_features:amount",
      "customer_transaction_features:hour",
      "customer_transaction_features:num_tx_past_day",
  ] , entity_rows=[{"customer_id": i} for i in range(1, 6)]   
).to_dict()
```

### Day 44 
> Store MLflow's Admin Password in HashiCorp Vault

- Loging to the vault UI (usbing the vault token given to us)
- Create the Secret Engine (with version: 2)
- Create a secret in `mlflow` path & create `admin_password` secret with random password


### Day 45 (Hard)
> Authenticate MLflow to Vault via AppRole and Fix Its KV Policy

- Edit the ACL policy
```shell
path "secret/data/mlflow" {
  capabilities = ["read"]
}
```


```shell
export VAULT_ADDR=http://127.0.0.1:8200
export VAULT_TOKEN=$(cat /root/code/vault-root-token)

vault auth enable approle

vault write auth/approle/role/mlflow \
  token_policies=mlflow-reader \
  token_ttl=1h token_max_ttl=4h

```



### Day 46
> Author Data-Quality Expectations with Great Expectations


```python
suite.add_expectation(ge.ExpectTableColumnsToMatchSet (column_set=["amount","hour","num_tx_past_day","is_fraud"]))
```

```python
suite.add_expectation(ge.ExpectColumnValuesToBeBetween(column="amount" , min_value=0))
```

```python
suite.add_expectation(ge.ExpectColumnValuesToBeBetween(column="hour",min_value=0,max_value=23))
```

```python
suite.add_expectation(ge.ExpectColumnValuesToBeInSet(column="is_fraud", value_set=[0,1]))
```





### Day 47
> Debug a Failing Great Expectations Checkpoint

```python
# replace the 0 with -400 based on the result from the run in UI
ge.ExpectColumnValuesToBeBetween(column="amount", min_value=-400)
```


### Day 48
> Enforce a Data-Quality Checkpoint as a Blocking CI Gate


Go to the workflow in `.gitea/workflows/data-quality` , & add the below step

```yaml
- name: running the command 
  run: python3 -m src.gx_run
```

```shell
# commit & push
git add .
git commit -m "Added CI"
git push
```

### Day 49 (GOOD one )
> Secrets + Data-Quality Integration Capstone


Step 1 : create a mlflow_password inside vault UI


Step 2 : 
```shell
TOKEN=$(cat /root/code/vault-token)
PASSWORD=$(curl -sf -H "X-Vault-Token: $TOKEN" \
  "$VAULT_ADDR/v1/secret/data/mlflow" \
  | python3 -c "import json, sys; print(json.load(sys.stdin)['data']['data']['mlflow_password'])")
if [ -z "$PASSWORD" ]; then
  echo "::error::Empty password from Vault -- stage mlflow_password in secret/mlflow first"
  exit 1
fi
echo "::notice::Fetched MLflow password from Vault (len=${#PASSWORD})"

```

```shell
# commit & push
git add .
git commit -m "Added CI"
git push
```

- Create a PR & merge the PR
- Check the actions , It should be successfully ran
- login to MLFlow UI , go to model and add Alias as `Production`



### Day 50
> Create Docker Image for ML Training Environment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install scikit-learn pandas numpy joblib
COPY train.py /app/train.py
CMD ["python3" , "train.py"]
```
```shell
# build the image
docker build -t ml-trainer:v1 .
```
```shell
# to validate
docker run --rm ml-trainer:v1 python3 -c "import sklearn, pandas, numpy, joblib; print('OK')" prints OK
```



### Day 51
> Create Multi-Stage Docker Build for ML Serving

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir scikit-learn pandas numpy joblib flask
COPY train_model.py /app/train_model.py
RUN python3 /app/train_model.py



FROM  python:3.11-slim AS runtime
WORKDIR /app
RUN pip install --no-cache-dir scikit-learn numpy joblib flask
COPY --from=builder /app/model.pkl /app/model.pkl
COPY serve.py /app/serve.py
EXPOSE 8080
CMD ["python3", "/app/serve.py"]

```

```shell
# build the image
docker build -t ml-serve:v1 .
```


```shell
# Validate it 
docker images ml-serve:v1 lists the built image; docker run --rm -p 8090:8080 ml-serve:v1

# On new tab
curl localhost:8090/health
```



### Day 52
> Fix a Broken Jupyter + MLflow + SeaweedFS Compose Stack

```yaml
jupyter
  command: "start-notebook.sh --ServerApp.token='' --ServerApp.password=''"
seaweedfs
      - "9000:8333" # S3 API
      - "9001:8888" # Filer UI
```

```shell
docker compose up -d 
docker compose ps
```

```
verify by opening all the 3 UI
```



### Day 53
> Fix a Broken PyTorch Dockerfile (CPU-Wheel URL)


```dockerfile
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

CMD ["python3", "-c", "import torch; print(torch.__version__, 'cuda?', torch.cuda.is_available())"]

```


```shell
# build the image
docker build -t dl-trainer:v1 .
```

```shell
# list the image
docker images dl-trainer:v1
```

```shell
# verify the image
docker run --rm dl-trainer:v1
```



### Day 54
> Push ML Model Images to Container Registry

```shell
REGISTRY="localhost:5555"
docker tag "$IMAGE" "$REGISTRY/$IMAGE"
docker push "$REGISTRY/$IMAGE"
```

```shell
# validate 
root@controlplane ~/code/ml-registry via 🐍 v3.12.3 ➜ curl http://localhost:5555/v2/_catalog
{"repositories":["fraud-detector"]}

root@controlplane ~/code/ml-registry via 🐍 v3.12.3 ➜  curl http://localhost:5555/v2/fraud-detector/tags/list
{"name":"fraud-detector","tags":["v1"]}

```


### Day 55
> Fix a Broken Dockerfile HEALTHCHECK and EXPOSE

```Dockerfile
EXPOSE 8085

HEALTHCHECK --interval=5s --timeout=3s --start-period=3s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8085/health')" || exit 1

```

```shell
docker build -t ml-health:v1 .

docker inspect --format '{{.Config.ExposedPorts}}' ml-health:v1

docker run ml-health:v1

```

### Day 56
> Fix a Docker CI Pipeline with Git-SHA Tagging

```shell 
# build.sh
REGISTRY="localhost:5555"


# --- Stage 1: test
python3 -m pytest app/test_app.py


# --- Stage 3: tag with short git SHA
SHA=$(git -C app rev-parse --short HEAD)
TAGGED="$REGISTRY/$IMAGE:$SHA"
```


### Day 57
> Serve an ML Model with Flask

```python
# appy.py

payload = request.get_json() or {}
amount = float(payload.get("amount", 0.0))
hour = int(payload.get("hour", 0))
num_tx_past_day = int(payload.get("num_tx_past_day", 0))
features = np.array([[amount, hour, num_tx_past_day]])
is_fraud = int(MODEL.predict(features)[0])
return jsonify({"is_fraud": is_fraud}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8085)
```



### Day 58
> Serve an ML Model with FastAPI

```python
# TODO 1
amount: float
hour: int = Field(ge=0, le=23)
num_tx_past_day: int = Field(ge=0)


# TODO 2
feature = [[req.amount, req.hour, req.num_tx_past_day]]
is_fraud = int(MODEL.predict(feature)[0])
prediction_history.append({
    "amount": req.amount, 
    "hour": req.hour, 
    "num_tx_past_day": req.num_tx_past_day,
    "is_fraud": is_fraud
})
return PredictResponse(is_fraud=is_fraud)
```

```shell
# To run the fastAPI server
uvicorn app:app --host 0.0.0.0 --port 8085
```

### Day 59
> Run Batch Predictions on a Dataset

```python
model = joblib.load(MODEL_PATH)

df = pd.read_csv(INPUT_CSV)
features = df[["amount", "hour", "num_tx_past_day"]]

df["prediction"] = model.predict(features)

df.to_csv(OUTPUT_CSV,index=False)
print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")

```

### Day 60
> Package a Model as a BentoML Service

```python
features = np.array([[amount, hour, num_tx_past_day]])
is_fraud = int(self.model.predict(features)[0])
self._history.append({
    "amount": amount, 
    "hour": hour, 
    "num_tx_past_day": num_tx_past_day,
    "is_fraud": is_fraud
})
return {"is_fraud": is_fraud}
```

```shell
bentoml serve service:FraudService
```

### Day 61
> Deploy a Model-Serving Container via Portainer


