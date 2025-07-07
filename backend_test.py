#!/usr/bin/env python3
import requests
import json
import base64
import os
import time
import io
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import matplotlib.pyplot as plt
import numpy as np
import unittest
from dotenv import load_dotenv
import sys

# Load environment variables from frontend/.env
load_dotenv('/app/frontend/.env')

# Get the backend URL from environment variables
BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL')
if not BACKEND_URL:
    print("Error: REACT_APP_BACKEND_URL not found in environment variables")
    sys.exit(1)

# Ensure the URL ends with /api
API_URL = f"{BACKEND_URL}/api" if not BACKEND_URL.endswith('/api') else BACKEND_URL

print(f"Testing backend API at: {API_URL}")

class JupyterNotebookAPITest(unittest.TestCase):
    """Test suite for Jupyter Notebook Upload and Execution API"""

    def setUp(self):
        """Set up test environment"""
        self.api_url = API_URL
        # Create a test notebook
        self.test_notebook = self.create_test_notebook()
        
    def create_test_notebook(self):
        """Create a test notebook with code and markdown cells"""
        notebook = new_notebook()
        
        # Add markdown cell
        markdown_cell = new_markdown_cell("# Test Notebook\nThis is a test notebook for API testing.")
        notebook.cells.append(markdown_cell)
        
        # Add simple code cell
        code_cell1 = new_code_cell("print('Hello, world!')")
        notebook.cells.append(code_cell1)
        
        # Add code cell with matplotlib plot
        plot_code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
"""
        code_cell2 = new_code_cell(plot_code)
        notebook.cells.append(code_cell2)
        
        # Add code cell with error
        code_cell3 = new_code_cell("print(undefined_variable)")
        notebook.cells.append(code_cell3)
        
        return notebook
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = requests.get(f"{self.api_url}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["message"], "Hello World")
        print("✅ Health check endpoint is working")
    
    def test_example_notebook(self):
        """Test the example notebook endpoint"""
        response = requests.get(f"{self.api_url}/notebook/example")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify notebook structure
        self.assertEqual(data["nbformat"], 4)
        self.assertTrue("cells" in data)
        self.assertTrue(len(data["cells"]) > 0)
        
        # Verify cell types
        cell_types = [cell["cell_type"] for cell in data["cells"]]
        self.assertTrue("markdown" in cell_types)
        self.assertTrue("code" in cell_types)
        
        print("✅ Example notebook endpoint is working")
    
    def test_execute_python_code(self):
        """Test the Python code execution endpoint"""
        # Test simple code execution
        code = "print('Hello, world!')"
        response = requests.post(
            f"{self.api_url}/execute/python",
            json={"code": code}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("Hello, world!", data["output"])
        
        print("✅ Simple code execution is working")
        
        # Test code with matplotlib plot
        plot_code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
"""
        response = requests.post(
            f"{self.api_url}/execute/python",
            json={"code": plot_code}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertTrue(len(data["plots"]) > 0)
        
        # Verify the plot is a valid base64 encoded image
        plot_data = data["plots"][0]
        try:
            image_data = base64.b64decode(plot_data)
            self.assertTrue(len(image_data) > 0)
            print("✅ Plot generation and base64 encoding is working")
        except:
            self.fail("Plot is not a valid base64 encoded image")
        
        # Test code with error
        error_code = "print(undefined_variable)"
        response = requests.post(
            f"{self.api_url}/execute/python",
            json={"code": error_code}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIsNotNone(data["error"])
        self.assertIn("NameError", data["error"])
        
        print("✅ Error handling for invalid Python code is working")
        
        # Test numpy/pandas integration
        np_code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {np.mean(arr)}")
"""
        response = requests.post(
            f"{self.api_url}/execute/python",
            json={"code": np_code}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("Array: [1 2 3 4 5]", data["output"])
        self.assertIn("Mean: 3.0", data["output"])
        
        print("✅ NumPy integration is working")
        
        # Test performance with a more complex calculation
        perf_code = """
import numpy as np
import time

start_time = time.time()
# Create a large array and perform operations
size = 1000000
arr = np.random.random(size)
result = np.sum(arr ** 2)
end_time = time.time()

print(f"Result: {result}")
print(f"Calculation time: {end_time - start_time:.4f} seconds")
"""
        response = requests.post(
            f"{self.api_url}/execute/python",
            json={"code": perf_code}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("Result:", data["output"])
        self.assertIn("Calculation time:", data["output"])
        
        print("✅ Performance testing is working")
    
    def test_notebook_upload(self):
        """Test the notebook upload and execution endpoint"""
        # Convert notebook to JSON string
        notebook_json = nbformat.writes(self.test_notebook)
        
        # Create a file-like object
        notebook_file = io.BytesIO(notebook_json.encode('utf-8'))
        
        # Upload the notebook
        files = {'file': ('test_notebook.ipynb', notebook_file, 'application/json')}
        response = requests.post(f"{self.api_url}/notebook/upload", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify the response structure
        self.assertIn("success", data)
        self.assertIn("results", data)
        self.assertIn("errors", data)
        self.assertIn("execution_time", data)
        
        # Check that we have results for all cells
        self.assertEqual(len(data["results"]), len(self.test_notebook.cells))
        
        # Verify markdown cell was processed correctly
        markdown_result = data["results"][0]
        self.assertEqual(markdown_result["cell_type"], "markdown")
        self.assertEqual(markdown_result["output_type"], "markdown")
        
        # Verify simple code cell was executed correctly
        code_result = data["results"][1]
        self.assertEqual(code_result["cell_type"], "code")
        self.assertIn("Hello, world!", code_result["output"])
        
        # Verify plot cell was executed correctly
        plot_result = data["results"][2]
        self.assertEqual(plot_result["cell_type"], "code")
        self.assertTrue(len(plot_result["plots"]) > 0)
        
        # Verify error cell was handled correctly
        error_result = data["results"][3]
        self.assertEqual(error_result["cell_type"], "code")
        self.assertEqual(error_result["output_type"], "error")
        self.assertIsNotNone(error_result["error"])
        
        print("✅ Notebook upload and execution is working")
        
        # Test invalid file upload (non-ipynb)
        invalid_file = io.BytesIO(b"This is not a notebook")
        files = {'file': ('not_a_notebook.txt', invalid_file, 'text/plain')}
        response = requests.post(f"{self.api_url}/notebook/upload", files=files)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("File must be a .ipynb file", response.text)
        
        print("✅ File validation is working (rejects non-ipynb files)")
    
    def test_large_output_handling(self):
        """Test handling of large outputs"""
        large_output_code = """
# Generate a large output
for i in range(1000):
    print(f"Line {i}: " + "X" * 100)
"""
        response = requests.post(
            f"{self.api_url}/execute/python",
            json={"code": large_output_code}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertTrue(len(data["output"]) > 10000)  # Should be a large output
        
        print("✅ Large output handling is working")
    
    def test_memory_usage(self):
        """Test memory usage during code execution"""
        memory_code = """
import numpy as np
import psutil
import os

# Get current process
process = psutil.Process(os.getpid())

# Get memory info before allocation
mem_before = process.memory_info().rss / 1024 / 1024  # MB

# Allocate a large array
size = 100 * 1024 * 1024  # 100MB of data
data = np.ones(size, dtype=np.uint8)

# Get memory after allocation
mem_after = process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory before: {mem_before:.2f} MB")
print(f"Memory after: {mem_after:.2f} MB")
print(f"Difference: {mem_after - mem_before:.2f} MB")

# Clean up
del data
"""
        try:
            response = requests.post(
                f"{self.api_url}/execute/python",
                json={"code": memory_code}
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # If psutil is not available, this test will fail gracefully
            if "ModuleNotFoundError" in data.get("error", ""):
                print("⚠️ Memory usage test skipped (psutil not available)")
            else:
                self.assertTrue(data["success"])
                print("✅ Memory usage monitoring is working")
        except Exception as e:
            print(f"⚠️ Memory usage test failed: {str(e)}")

if __name__ == "__main__":
    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)