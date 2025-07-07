from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
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
import torch
import numpy as np
from PIL import Image
import cv2
import imageio
import gradio as gr
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import tempfile
import asyncio
import shutil


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Get HuggingFace token
hf_token = os.environ.get('HF_TOKEN')

# Global variables for video generation models
sd_pipe = None
svd_pipe = None
gradio_app = None

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Video generation directory
VIDEO_DIR = ROOT_DIR / "generated_videos"
VIDEO_DIR.mkdir(exist_ok=True)


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

class VideoGenerationRequest(BaseModel):
    prompt: str = "realistic beautiful woman walking, 6 seconds, high quality"
    negative_prompt: str = "blurry, low quality, distorted, cartoon"
    num_frames: int = 144  # 6 seconds * 24 fps
    fps: int = 24
    width: int = 1280
    height: int = 720
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02

class VideoGenerationResponse(BaseModel):
    success: bool
    video_path: str = None
    error: str = None
    generation_time: float = 0.0
    frames_generated: int = 0

class GradioLaunchResponse(BaseModel):
    success: bool
    gradio_url: str = None
    error: str = None

# Video generation helper functions
def initialize_models():
    """Initialize video generation models"""
    global sd_pipe, svd_pipe
    
    try:
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
            
        print(f"Initializing models on {device}")
        
        # Initialize Stable Diffusion for image generation
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            use_auth_token=hf_token,
            safety_checker=None,
            requires_safety_checker=False
        )
        sd_pipe = sd_pipe.to(device)
        sd_pipe.enable_model_cpu_offload()
        
        # Initialize Stable Video Diffusion
        svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
            torch_dtype=dtype,
            use_auth_token=hf_token,
            variant="fp16" if dtype == torch.float16 else None
        )
        svd_pipe = svd_pipe.to(device)
        svd_pipe.enable_model_cpu_offload()
        
        print("Models initialized successfully")
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        sd_pipe = None
        svd_pipe = None

def generate_initial_image(prompt: str, negative_prompt: str = None) -> Image.Image:
    """Generate initial image using Stable Diffusion"""
    if sd_pipe is None:
        raise ValueError("Stable Diffusion model not initialized")
    
    enhanced_prompt = f"realistic, highly detailed, beautiful young woman, {prompt}, professional photography, 8k, ultra realistic"
    
    image = sd_pipe(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        width=1280,
        height=720,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]
    
    return image

def generate_video_from_image(
    image: Image.Image,
    num_frames: int = 144,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    fps: int = 24
) -> str:
    """Generate video from initial image using SVD"""
    if svd_pipe is None:
        raise ValueError("Stable Video Diffusion model not initialized")
    
    # Generate video frames
    frames = svd_pipe(
        image=image,
        num_frames=num_frames,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        decode_chunk_size=8
    ).frames[0]
    
    # Save video
    video_path = VIDEO_DIR / f"generated_video_{uuid.uuid4()}.mp4"
    export_to_video(frames, str(video_path), fps=fps)
    
    return str(video_path)

def create_gradio_interface():
    """Create Gradio interface for video generation"""
    def gradio_generate_video(prompt, negative_prompt, num_frames, fps):
        try:
            # Generate initial image
            image = generate_initial_image(prompt, negative_prompt)
            
            # Generate video
            video_path = generate_video_from_image(
                image=image,
                num_frames=int(num_frames),
                fps=int(fps)
            )
            
            return video_path, "Video generated successfully!"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=gradio_generate_video,
        inputs=[
            gr.Textbox(
                label="Prompt", 
                value="realistic beautiful woman walking in park, natural lighting",
                placeholder="Describe the video you want to generate..."
            ),
            gr.Textbox(
                label="Negative Prompt", 
                value="blurry, low quality, distorted, cartoon, anime",
                placeholder="What you don't want in the video..."
            ),
            gr.Slider(
                minimum=25,
                maximum=144,
                value=144,
                step=1,
                label="Number of Frames (144 = 6 seconds at 24fps)"
            ),
            gr.Slider(
                minimum=12,
                maximum=30,
                value=24,
                step=1,
                label="FPS (Frames Per Second)"
            )
        ],
        outputs=[
            gr.Video(label="Generated Video"),
            gr.Textbox(label="Status")
        ],
        title="🎬 AI Video Generator",
        description="Generate realistic human female videos using Stable Video Diffusion. Optimized for T4 GPU.",
        examples=[
            ["realistic beautiful woman walking confidently, elegant dress, natural lighting", "blurry, low quality", 144, 24],
            ["attractive woman sitting in cafe, smiling, professional lighting", "cartoon, anime, low quality", 120, 24],
            ["beautiful woman dancing gracefully, flowing hair, studio lighting", "distorted, blurry, artifacts", 144, 24]
        ]
    )
    
    return interface

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

@api_router.post("/notebook/upload", response_model=NotebookExecutionResult)
async def upload_and_execute_notebook(file: UploadFile = File(...)):
    """Upload and execute a Jupyter notebook"""
    if not file.filename.endswith('.ipynb'):
        raise HTTPException(status_code=400, detail="File must be a .ipynb file")
    
    try:
        content = await file.read()
        notebook_content = content.decode('utf-8')
        
        # Parse and execute the notebook
        result = parse_and_execute_notebook(notebook_content)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing notebook: {str(e)}")

@api_router.post("/execute/python", response_model=ExecutePythonCodeResponse)
async def execute_python_code_endpoint(request: ExecutePythonCodeRequest):
    """Execute Python code directly"""
    try:
        result = execute_python_code(request.code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing code: {str(e)}")

@api_router.get("/notebook/example")
async def get_example_notebook():
    """Get an example notebook for testing"""
    example_notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Example Notebook\n", "This is a sample notebook for testing purposes."]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": ["print('Hello from Jupyter!')"]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": ["import matplotlib.pyplot as plt\nimport numpy as np\n\n# Create a simple plot\nx = np.linspace(0, 10, 100)\ny = np.sin(x)\n\nplt.figure(figsize=(10, 6))\nplt.plot(x, y)\nplt.title('Sine Wave')\nplt.xlabel('X')\nplt.ylabel('Y')\nplt.grid(True)\nplt.show()"]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [],
                "source": ["# Simple calculation\nresult = 2 + 2\nprint(f'2 + 2 = {result}')"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return JSONResponse(content=example_notebook)

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
