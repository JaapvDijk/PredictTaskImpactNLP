from google.cloud import automl
from google.oauth2 import service_account

project_id = 'disco-plane-308520'
display_name = 'google_nlp'
region = 'us-central1'

credentials = service_account.Credentials.from_service_account_file("disco-plane-308520-c946f9c07b95.json")
client = automl.AutoMlClient(credentials=credentials)

project_location = f"projects/{project_id}/locations/{region}"

metadata = automl.TextClassificationDatasetMetadata(
    classification_type=automl.ClassificationType.MULTICLASS
)
dataset = automl.Dataset(
    display_name=display_name,
    text_classification_dataset_metadata=metadata,
)

response = client.create_dataset(parent=project_location, dataset=dataset)

created_dataset = response.result()

print("Dataset name: {}".format(created_dataset.name))
print("Dataset id: {}".format(created_dataset.name.split("/")[-1]))