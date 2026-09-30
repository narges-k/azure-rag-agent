"""Step 1: upload the documents in ./docs to the Blob Storage container.
Use PUBLIC documents only (e.g. open reports, manuals, Wikipedia exports) - never company data."""
from pathlib import Path

from azure.core.exceptions import ResourceExistsError

from common import blob_service, env


def main() -> None:
    service = blob_service()
    container = service.get_container_client(env("AZURE_STORAGE_CONTAINER"))
    try:
        container.create_container()
        print(f"Created container {container.container_name}")
    except ResourceExistsError:
        pass

    files = [p for p in Path("docs").glob("**/*") if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    if not files:
        raise SystemExit("No .pdf/.txt/.md files found in ./docs - add some first.")
    for path in files:
        with open(path, "rb") as f:
            container.upload_blob(name=path.name, data=f, overwrite=True)
        print("uploaded", path.name)


if __name__ == "__main__":
    main()
