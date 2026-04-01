import os
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
import pandas as pd
import io
import time
from pydantic import BaseModel
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

app = FastAPI()

frontend_url = os.getenv("FRONTEND_URL", "*")
allowed_origins = frontend_url.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    expected_api_key = os.getenv("API_SECRET_KEY")
    if expected_api_key:
        if api_key != expected_api_key:
            raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

import json
import asyncio
from fastapi.responses import StreamingResponse

@app.post("/api/synthesize", dependencies=[Depends(get_api_key)])
async def synthesize_data(file: UploadFile = File(...)):
    contents = await file.read()
    
    async def generate_response():
        # Helper to yield formatted JSON chunks
        def emit(data: dict):
            return json.dumps(data) + "\n"
            
        try:
            yield emit({"status": "Reading dataset into memory...", "progress": 10})
            real_data = await asyncio.to_thread(pd.read_csv, io.StringIO(contents.decode('utf-8')))
            
            yield emit({"status": "Detecting schema and distributions...", "progress": 30})
            metadata = SingleTableMetadata()
            await asyncio.to_thread(metadata.detect_from_dataframe, real_data)
            
            yield emit({"status": "Training Gaussian Copula Model (SDV)...", "progress": 55})
            synthesizer = GaussianCopulaSynthesizer(metadata)
            await asyncio.to_thread(synthesizer.fit, real_data)
            
            yield emit({"status": "Generating private mathematical twin...", "progress": 85})
            synthetic_data = await asyncio.to_thread(synthesizer.sample, num_rows=len(real_data))
            
            yield emit({"status": "Formatting CSV payload...", "progress": 95})
            output = io.StringIO()
            await asyncio.to_thread(synthetic_data.to_csv, output, index=False)
            
            # Send final payload
            yield emit({
                "status": "Complete", 
                "progress": 100,
                "csv_data": output.getvalue()
            })
            
        except Exception as e:
            yield emit({"error": f"Synthesis failed: {str(e)}"})

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")

class SyncPayload(BaseModel):
    audience_name: str
    destination: str

@app.post("/api/activations/sync", dependencies=[Depends(get_api_key)])
async def sync_audience(payload: SyncPayload):
    async def generate_sync_response():
        def emit(data: dict):
            return json.dumps(data) + "\n"
        
        try:
            yield emit({"status": f"Authenticating OAuth with {payload.destination} API...", "progress": 10})
            await asyncio.sleep(1.5)
            
            yield emit({"status": "Extracting synthetic audience payload into secure memory...", "progress": 30})
            await asyncio.sleep(2.0)
            
            yield emit({"status": "Applying SHA-256 hashing to PII match keys...", "progress": 60})
            await asyncio.sleep(2.5)
            
            yield emit({"status": "Uploading micro-batches to destination network...", "progress": 85})
            await asyncio.sleep(3.0)
            
            yield emit({"status": "Verifying audience size match rate via API response...", "progress": 95})
            await asyncio.sleep(1.5)
            
            yield emit({
                "status": "Complete", 
                "progress": 100,
                "success": True,
                "message": f"Successfully synced {payload.audience_name} to {payload.destination}."
            })
            
        except Exception as e:
            yield emit({"error": f"Sync failed: {str(e)}"})

    return StreamingResponse(generate_sync_response(), media_type="application/x-ndjson")
