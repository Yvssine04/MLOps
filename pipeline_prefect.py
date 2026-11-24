import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, get_run_logger, task

from main import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)
from model_pipeline import predict

PYTHON = sys.executable
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_command(command):
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(command))
    result = subprocess.run(  # nosec B603
        command, capture_output=True, text=True, check=False
    )
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        logger.error(result.stderr.strip())
        raise RuntimeError(f"Echec : {' '.join(command)}")



@task(name="git-pull")
def pull_task():
    run_command(["git", "pull"])

# 1) Installer les dependances
@task(name="install-dependencies")
def install_task():
    run_command([PYTHON, "-m", "pip", "install", "-q", "-r", "requirements.txt"])


# Etapes liees au code
@task(name="format-code")
def format_task():
    run_command([PYTHON, "-m", "black", *CODE_FILES])


@task(name="code-quality")
def quality_task():
    run_command([PYTHON, "-m", "flake8", "--max-line-length=88", *CODE_FILES])


@task(name="code-security")
def security_task():
    run_command([PYTHON, "-m", "bandit", "-ll", *CODE_FILES])


@task(name="unit-tests")
def tests_task():
    run_command([PYTHON, "-m", "pytest", "-q", "tests"])


# 2) Preparation des donnees
@task(name="prepare-data")
def prepare_task():
    return prepare_data()


# 3) Entrainement du modele
@task(name="train-model")
def train_task(data):
    x_train, x_test, y_train, y_test = data
    return train_model(x_train, y_train)


# 4) Sauvegarder le modele
@task(name="save-model")
def save_task(model):
    save_model(model)


# 5) Charger le modele
@task(name="load-model")
def load_task():
    return load_model()


# 6) Evaluer le modele
@task(name="evaluate-model")
def evaluate_task(model, data=None):
    if data is None:
        data = prepare_data()
    x_train, x_test, y_train, y_test = data
    return evaluate_model(model, x_test, y_test)


# 7) Prediction
@task(name="predict")
def predict_task():
    x_train, x_test, y_train, y_test = prepare_data()
    predictions = predict(x_test)
    print(f"Clients predits partants : {int(predictions.sum())} / {len(predictions)}")
    return predictions


@flow(name="code", log_prints=True)
def flow_code():
    format_task()
    quality_task()
    security_task()
    tests_task()


@flow(name="train", log_prints=True)
def flow_train():
    data = prepare_task()
    train_task(data)


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    model = load_task()
    evaluate_task(model)


@flow(name="predict", log_prints=True)
def flow_predict():
    predict_task()


@flow(name="all", log_prints=True)
def flow_all():
    pull_task()
    install_task()
    format_task()
    quality_task()
    security_task()
    tests_task()
    data = prepare_task()
    model = train_task(data)
    save_task(model)
    evaluate_task(model, data)


FLOWS = {
    "all": flow_all,
    "train": flow_train,
    "entrainement": flow_train,
    "evaluate": flow_evaluate,
    "predict": flow_predict,
    "code": flow_code,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
