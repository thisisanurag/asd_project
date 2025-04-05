import React, { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [problemStatement, setProblemStatement] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const [annotatedImage, setAnnotatedImage] = useState('');
  const [gptFeedback, setGptFeedback] = useState("");
  const [loading, setLoading] = useState(false);

  // Read the backend API URL from environment variables
  const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setFileName(file.name);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setStatusMessage("Please select a file first.");
      return;
    }
    
    setLoading(true);
    setStatusMessage("");
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('problem_statement', problemStatement || "Analyze this circuit for completeness and accuracy.");

    try {
      const response = await axios.post(`${API_URL}/upload` formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setStatusMessage(response.data.message);
      setAnnotatedImage(response.data.annotated_image);
      setGptFeedback(response.data.gpt_feedback);
    } catch (error) {
      console.error(error);
      setStatusMessage("Error uploading image");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <header className="hero-section">
        <div className="hero-overlay">
          <h1 className="hero-title">Circuit Grader</h1>
          <p className="hero-subtitle">
            Upload your circuit diagram and get an instant professional grading report.
          </p>
        </div>
      </header>

      <main className="main-content">
        <div className="upload-card">
          <h2 className="card-title">Upload Your Circuit</h2>
          <p className="card-description">
            Select an image file and describe your circuit below before clicking "Upload & Grade".
          </p>

          <textarea
            className="problem-input"
            placeholder="Enter your problem statement..."
            rows="4"
            value={problemStatement}
            onChange={(e) => setProblemStatement(e.target.value)}
          ></textarea>

          <label className="custom-file-upload">
            <input
              type="file"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            Choose File
          </label>

          {fileName && <span className="file-name">{fileName}</span>}

          <button className="upload-button" onClick={handleUpload}>
            Upload &amp; Grade
          </button>

          {statusMessage && <p className="status-message">{statusMessage}</p>}
        </div>

        {loading && (
          <div className="loader-container">
            <div className="loader"></div>
            <p>Loading, please wait...</p>
          </div>
        )}

        {annotatedImage && !loading && (
          <div className="result-section">
            <h3>Annotated Image</h3>
            <img
              className="annotated-image"
              src={annotatedImage}
              alt="Annotated Circuit"
            />
          </div>
        )}

        {gptFeedback && !loading && (
          <div className="feedback-section">
            <h3>AI-Powered Circuit Grading Report</h3>
            <div className="gpt-feedback">
              <ReactMarkdown 
                remarkPlugins={[remarkMath]} 
                rehypePlugins={[rehypeKatex]}
              >
                {gptFeedback}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </main>

      <footer className="site-footer">
        <p>&copy; {new Date().getFullYear()} Genesis Creative Collective Inspired</p>
      </footer>
    </div>
  );
}

export default App;
