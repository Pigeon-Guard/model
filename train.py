from ultralytics import YOLO
from ultralytics.data.split import autosplit
import wandb, argparse
wandb.login()

EXTRACTED_CONFIG: dict = {}

def extract_config(trainer):
    EXTRACTED_CONFIG['run_name'] = trainer.args.name
    EXTRACTED_CONFIG['run_id'] = wandb.run.id

def callback_fit_epoch_end(trainer):
    run = wandb.run
    epoch = trainer.epoch
    fitness_dict = {}
    if trainer.fitness is not None:
        fitness_dict = {"train/fitness": float(trainer.fitness)}

    run.log(
        {**fitness_dict},
        step=epoch,
    )

def callback_train_start(trainer):
    run = wandb.run
    if run is None:
        raise RuntimeError("W&B run is not initialized")

    run.log({"ci_test/start_flag": 1})

def main(args):
    run = wandb.init(
        project="ultralytics",
        name="yolov8n",
        job_type="train",
    )

    autosplit(
        path="dataset/images",
        weights=(0.8, 0.1, 0.1),
    )

    model = YOLO("yolov8n.pt")

    model.add_callback("on_train_start", callback_train_start)
    model.add_callback("on_fit_epoch_end", callback_fit_epoch_end)

    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)

    metrics = model.val(data=args.data)
    latest_fitness = metrics.results_dict["fitness"]

    wandb_api = wandb.Api()
    new_best = False

    try:
        current_best_artifact = wandb_api.artifact(
            "wandb-registry-pigeon-guard/yolo-model:production"
        )

        current_best_fitness = current_best_artifact.metadata.get("validation_fitness")
        print(f"Current best fitness: {current_best_fitness}")
        if latest_fitness >= current_best_fitness:
            new_best = True
            print("New best model found")

    except wandb.errors.CommError:
        print("No existing best artifact found")
        new_best = True

    if new_best:
        print("new best")
        # artifact = wandb_api.artifact(f"pigeon-guard/ultralytics/run_{run.id}_model:best")
        # artifact.metadata["validation_fitness"] = latest_fitness
        # artifact.link("wandb-registry-pigeon-guard/yolo-model", aliases=["production"])
        # artifact.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=224)
    main(parser.parse_args())
