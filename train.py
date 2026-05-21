from ultralytics import YOLO
from ultralytics.data.split import autosplit
import wandb, argparse
wandb.login()

def callback_fit_epoch_end(trainer):
    run = wandb.run
    metrics_dict = {name: value for name, value in trainer.metrics.items()}
    fitness_dict = {"metrics/fitness": trainer.fitness}
    run.log({**metrics_dict, **fitness_dict})

def callback_train_epoch_end(trainer):
    run = wandb.run
    train_dict = {name: value for name, value in trainer.label_loss_items(trainer.tloss).items()}
    run.log(train_dict)

def callback_train_end(trainer):
    run = wandb.run
    artifact = wandb.Artifact(
        name=f"model-{wandb.run.id}",
        type="model",
        description="Best checkpoint",
        metadata={"best_fitness": trainer.best_fitness, **trainer.metrics},
    )
    artifact.add_file(str(trainer.best), name="best.pt")
    run.log_artifact(artifact, aliases=["best"])

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

    model.add_callback("on_train_epoch_end", callback_train_epoch_end)
    model.add_callback("on_fit_epoch_end", callback_fit_epoch_end)
    model.add_callback("on_train_end", callback_train_end)

    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, plots=False)

    metrics = model.val(data=args.data, split="test")
    latest_fitness = metrics.results_dict["fitness"]

    wandb_api = wandb.Api()
    new_best = False

    # Wait for upload of artifact
    import time
    time.sleep(60)

    try:
        current_best_artifact = wandb_api.artifact(
            "wandb-registry-pigeon-guard/yolo-model:production"
        )

        current_best_fitness = current_best_artifact.metadata.get("test-fitness", 0)
        print(f"Current best fitness: {current_best_fitness}")
        if latest_fitness >= current_best_fitness:
            new_best = True
            print("New best model found")

    except wandb.errors.CommError:
        print("No existing best artifact found")
        new_best = True

    if new_best:
        artifact = wandb_api.artifact(f"pigeon-guard/ultralytics/model-{run.id}:best")
        artifact.metadata["test-fitness"] = latest_fitness
        artifact.link("wandb-registry-pigeon-guard/yolo-model", aliases=["production"])
        artifact.save()
        print("Best model updated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=224)
    main(parser.parse_args())
