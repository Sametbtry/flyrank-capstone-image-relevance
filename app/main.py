from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {
        "project_name" : "AI Image Understanding & Content Matching Engine",
    }


@app.get("/health")
def get_health():
    return {"status": "ok"}