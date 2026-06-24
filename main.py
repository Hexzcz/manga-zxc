import uvicorn
from multiprocessing import Manager
from pipeline import start_pipeline
import server


if __name__ == "__main__":
    manager = Manager()

    pipeline_ctx = start_pipeline(manager)

    server.inject_pipeline(pipeline_ctx)

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )