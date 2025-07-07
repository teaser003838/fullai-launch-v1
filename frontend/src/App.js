import React, { useEffect, useState } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const NotebookUpload = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResults(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a .ipynb file");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(`${API}/notebook/upload`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Error uploading notebook");
    } finally {
      setLoading(false);
    }
  };

  const downloadExampleNotebook = async () => {
    try {
      const response = await axios.get(`${API}/notebook/example`);
      const blob = new Blob([JSON.stringify(response.data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "example.ipynb";
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError("Error downloading example notebook");
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-center">
          Jupyter Notebook Executor
        </h1>

        {/* Upload Section */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Upload & Execute Notebook</h2>
          
          <div className="space-y-4">
            <div>
              <input
                type="file"
                accept=".ipynb"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
              />
            </div>
            
            <div className="flex gap-4">
              <button
                onClick={handleUpload}
                disabled={loading || !file}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-6 py-2 rounded-lg font-semibold"
              >
                {loading ? "Executing..." : "Upload & Execute"}
              </button>
              
              <button
                onClick={downloadExampleNotebook}
                className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg font-semibold"
              >
                Download Example
              </button>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-6">
            <p className="text-red-200">{error}</p>
          </div>
        )}

        {/* Results Display */}
        {results && (
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Execution Results</h2>
            
            <div className="mb-4">
              <p className="text-sm text-gray-300">
                Success: {results.success ? "✅" : "❌"} | 
                Execution Time: {results.execution_time.toFixed(2)}s | 
                Errors: {results.errors.length}
              </p>
            </div>

            {results.errors.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-red-400 mb-2">Errors:</h3>
                <div className="bg-red-900 border border-red-700 rounded-lg p-4">
                  {results.errors.map((error, index) => (
                    <p key={index} className="text-red-200 mb-2">{error}</p>
                  ))}
                </div>
              </div>
            )}

            <div className="space-y-4">
              {results.results.map((cell, index) => (
                <div key={index} className="border border-gray-700 rounded-lg p-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-gray-400">
                      Cell {cell.execution_count} - {cell.cell_type}
                    </span>
                    <span className="text-sm text-gray-400">
                      {cell.output_type}
                    </span>
                  </div>
                  
                  {/* Source Code */}
                  <div className="mb-3">
                    <h4 className="text-sm font-semibold text-gray-300 mb-1">Source:</h4>
                    <pre className="bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                      <code>{cell.source}</code>
                    </pre>
                  </div>
                  
                  {/* Output */}
                  {cell.output && (
                    <div className="mb-3">
                      <h4 className="text-sm font-semibold text-gray-300 mb-1">Output:</h4>
                      <pre className="bg-gray-700 p-3 rounded text-sm overflow-x-auto">
                        <code>{cell.output}</code>
                      </pre>
                    </div>
                  )}
                  
                  {/* Plots */}
                  {cell.plots && cell.plots.length > 0 && (
                    <div className="mb-3">
                      <h4 className="text-sm font-semibold text-gray-300 mb-1">Plots:</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {cell.plots.map((plot, plotIndex) => (
                          <img
                            key={plotIndex}
                            src={`data:image/png;base64,${plot}`}
                            alt={`Plot ${plotIndex + 1}`}
                            className="rounded-lg max-w-full h-auto"
                          />
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Errors */}
                  {cell.error && (
                    <div className="mb-3">
                      <h4 className="text-sm font-semibold text-red-400 mb-1">Error:</h4>
                      <pre className="bg-red-900 p-3 rounded text-sm overflow-x-auto">
                        <code>{cell.error}</code>
                      </pre>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const CodeExecutor = () => {
  const [code, setCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleExecute = async () => {
    if (!code.trim()) {
      setError("Please enter some Python code");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API}/execute/python`, {
        code: code,
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Error executing code");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-center">
          Python Code Executor
        </h1>

        <div className="bg-gray-800 rounded-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Execute Python Code</h2>
          
          <div className="space-y-4">
            <div>
              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="Enter your Python code here..."
                className="w-full h-64 p-4 bg-gray-900 border border-gray-700 rounded-lg text-white font-mono text-sm"
              />
            </div>
            
            <button
              onClick={handleExecute}
              disabled={loading || !code.trim()}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-6 py-2 rounded-lg font-semibold"
            >
              {loading ? "Executing..." : "Execute Code"}
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-6">
            <p className="text-red-200">{error}</p>
          </div>
        )}

        {result && (
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Execution Result</h2>
            
            <div className="mb-4">
              <p className="text-sm text-gray-300">
                Success: {result.success ? "✅" : "❌"}
              </p>
            </div>

            {result.output && (
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-green-400 mb-2">Output:</h3>
                <pre className="bg-gray-700 p-4 rounded text-sm overflow-x-auto">
                  <code>{result.output}</code>
                </pre>
              </div>
            )}

            {result.plots && result.plots.length > 0 && (
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-blue-400 mb-2">Plots:</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {result.plots.map((plot, index) => (
                    <img
                      key={index}
                      src={`data:image/png;base64,${plot}`}
                      alt={`Plot ${index + 1}`}
                      className="rounded-lg max-w-full h-auto"
                    />
                  ))}
                </div>
              </div>
            )}

            {result.error && (
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-red-400 mb-2">Error:</h3>
                <pre className="bg-red-900 p-4 rounded text-sm overflow-x-auto">
                  <code>{result.error}</code>
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const Home = () => {
  const helloWorldApi = async () => {
    try {
      const response = await axios.get(`${API}/`);
      console.log(response.data.message);
    } catch (e) {
      console.error(e, `errored out requesting / api`);
    }
  };

  useEffect(() => {
    helloWorldApi();
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <a
            href="https://emergent.sh"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block mb-6"
          >
            <img 
              src="https://avatars.githubusercontent.com/in/1201222?s=120&u=2686cf91179bbafbc7a71bfbc43004cf9ae1acea&v=4" 
              alt="Emergent Logo"
              className="w-16 h-16 rounded-full"
            />
          </a>
          <h1 className="text-4xl font-bold mb-4">Jupyter Notebook Executor</h1>
          <p className="text-xl text-gray-300">Upload and execute .ipynb files with ease</p>
        </header>

        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-2xl font-semibold mb-4">📓 Upload Notebook</h2>
            <p className="text-gray-300 mb-6">
              Upload your .ipynb file and execute all cells with comprehensive output display
            </p>
            <Link 
              to="/notebook"
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold inline-block"
            >
              Upload & Execute
            </Link>
          </div>

          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-2xl font-semibold mb-4">💻 Execute Code</h2>
            <p className="text-gray-300 mb-6">
              Write and execute Python code directly in the browser with plot support
            </p>
            <Link 
              to="/code"
              className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg font-semibold inline-block"
            >
              Execute Code
            </Link>
          </div>
        </div>

        <div className="mt-12 text-center">
          <h3 className="text-2xl font-semibold mb-4">Features</h3>
          <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            <div className="text-center">
              <div className="text-4xl mb-2">🚀</div>
              <h4 className="font-semibold mb-2">Fast Execution</h4>
              <p className="text-sm text-gray-300">Execute notebooks and code with minimal latency</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">📊</div>
              <h4 className="font-semibold mb-2">Plot Support</h4>
              <p className="text-sm text-gray-300">Matplotlib plots are captured and displayed</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">🔧</div>
              <h4 className="font-semibold mb-2">Error Handling</h4>
              <p className="text-sm text-gray-300">Comprehensive error reporting and debugging</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/notebook" element={<NotebookUpload />} />
          <Route path="/code" element={<CodeExecutor />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
