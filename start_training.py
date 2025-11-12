# start_training.py
import os
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.inputs import TrainingInput

# ----- CONFIGURE THESE -----
ROLE_ARN = os.getenv("SM_ROLE_ARN")          # set as GitHub secret
BUCKET = "prathvi-raw"                       # your bucket
PROCESSED_PREFIX = "processed/"              # input data prefix
OUTPUT_S3 = f"s3://{BUCKET}/models/"         # where model artifacts go
FRAMEWORK_VERSION = "1.2-1"                  # sklearn container
INSTANCE_TYPE = "ml.m5.large"                # training instance
# ---------------------------

session = sagemaker.Session()

# estimator will run train.py from the repo root
estimator = SKLearn(
    entry_point="train.py",
    source_dir=".",                  # include your script from repo root
    role=ROLE_ARN,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    framework_version=FRAMEWORK_VERSION,
    py_version="py3",
    output_path=OUTPUT_S3,
    sagemaker_session=session,
    base_job_name="etl-ml-diabetes"
)

# point to your processed CSVs in S3
train_input = TrainingInput(
    s3_data=f"s3://{BUCKET}/{PROCESSED_PREFIX}",
    content_type="text/csv"
)

print("Starting training job…")
estimator.fit({"train": train_input}, wait=True, logs=True)
print("Training complete.")
print(f"Model artifact at: {estimator.model_data}")
