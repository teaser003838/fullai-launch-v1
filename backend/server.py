from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid
from datetime import datetime
import json
import nbformat
from nbconvert import PythonExporter
import io
import sys
import traceback
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class NotebookExecutionResult(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    errors: List[str]
    execution_time: float

class CellOutput(BaseModel):
    cell_type: str
    source: str
    output: str
    output_type: str
    execution_count: int
    error: str = None
    plots: List[str] = []  # Base64 encoded plots

class ExecutePythonCodeRequest(BaseModel):
    code: str

class ExecutePythonCodeResponse(BaseModel):
    success: bool
    output: str
    error: str = None
    plots: List[str] = []

# Helper functions
def capture_plots():
    """Capture matplotlib plots as base64 encoded images"""
    plots = []
    for i in plt.get_fignums():
        fig = plt.figure(i)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode()
        plots.append(plot_data)
        plt.close(fig)
    return plots

def execute_python_code(code: str) -> ExecutePythonCodeResponse:
    """Execute Python code and capture output"""
    # Create StringIO objects to capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Redirect stdout and stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        # Create a new namespace for execution
        namespace = {
            '__builtins__': __builtins__,
            'plt': plt,
            'numpy': None,
            'pandas': None,
            'matplotlib': matplotlib
        }
        
        # Try to import common packages
        try:
            import numpy as np
            import pandas as pd
            namespace['np'] = np
            namespace['pd'] = pd
            namespace['numpy'] = np
            namespace['pandas'] = pd
        except ImportError:
            pass
        
        # Execute the code
        exec(code, namespace)
        
        # Capture any plots
        plots = capture_plots()
        
        # Get the output
        output = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()
        
        return ExecutePythonCodeResponse(
            success=True,
            output=output,
            error=error_output if error_output else None,
            plots=plots
        )
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return ExecutePythonCodeResponse(
            success=False,
            output=stdout_capture.getvalue(),
            error=error_msg
        )
    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def parse_and_execute_notebook(notebook_content: str) -> NotebookExecutionResult:
    """Parse and execute a Jupyter notebook"""
    try:
        # Parse the notebook
        notebook = nbformat.reads(notebook_content, as_version=4)
        
        results = []
        errors = []
        start_time = datetime.now()
        
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                # Execute the cell
                code = cell.source
                execution_result = execute_python_code(code)
                
                cell_result = CellOutput(
                    cell_type=cell.cell_type,
                    source=code,
                    output=execution_result.output,
                    output_type="execute_result" if execution_result.success else "error",
                    execution_count=i + 1,
                    error=execution_result.error,
                    plots=execution_result.plots
                )
                
                results.append(cell_result.dict())
                
                if not execution_result.success:
                    errors.append(f"Cell {i+1}: {execution_result.error}")
            
            elif cell.cell_type == 'markdown':
                # Handle markdown cells
                cell_result = CellOutput(
                    cell_type=cell.cell_type,
                    source=cell.source,
                    output=cell.source,  # For markdown, output is the same as source
                    output_type="markdown",
                    execution_count=i + 1
                )
                results.append(cell_result.dict())
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return NotebookExecutionResult(
            success=len(errors) == 0,
            results=results,
            errors=errors,
            execution_time=execution_time
        )
        
    except Exception as e:
        return NotebookExecutionResult(
            success=False,
            results=[],
            errors=[f"Failed to parse notebook: {str(e)}"],
            execution_time=0.0
        )

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
