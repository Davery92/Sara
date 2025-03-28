import { useState, useEffect, useRef } from 'react';
import { Bars3Icon } from '@heroicons/react/24/outline';
import { ArrowPathIcon, DocumentIcon, TrashIcon, DocumentArrowDownIcon } from '@heroicons/react/24/outline';
import Sidebar from '../components/Sidebar';
import BriefingModal from '../components/BriefingModal';

const API_BASE_URL = 'http://10.185.1.8:7009';

const DocumentPage = () => {
  // State
  const [showSidebar, setShowSidebar] = useState(true);
  const [isMobile, setIsMobile] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');
  const [documents, setDocuments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [notification, setNotification] = useState({ message: '', type: '', visible: false });
  const [selectedFile, setSelectedFile] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [briefingModal, setBriefingModal] = useState({
    isOpen: false,
    title: '',
    filename: ''
  });
  
  // Form state
  const [documentTitle, setDocumentTitle] = useState('');
  const [documentTags, setDocumentTags] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  
  // Refs
  const fileInputRef = useRef(null);
  const uploadAreaRef = useRef(null);

  // Check if mobile on mount and window resize
  useEffect(() => {
    const checkIfMobile = () => {
      setIsMobile(window.innerWidth <= 768);
      if (window.innerWidth <= 768) {
        setShowSidebar(false);
      }
    };
    
    checkIfMobile();
    window.addEventListener('resize', checkIfMobile);
    
    return () => {
      window.removeEventListener('resize', checkIfMobile);
    };
  }, []);

  // Load documents when tab changes to 'documents'
  useEffect(() => {
    if (activeTab === 'documents') {
      loadDocuments();
    }
  }, [activeTab]);

  // Load documents function
  const loadDocuments = async () => {
    setIsLoading(true);
    
    try {
      // Try primary endpoint first
      const response = await fetch(`${API_BASE_URL}/rag/documents`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Cache-Control': 'no-cache'
        }
      });
      
      // Check content type before attempting to parse JSON
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        console.error('Non-JSON response received');
        // Try fallback endpoint
        return await loadDocumentsFromFallback();
      }
      
      if (!response.ok) {
        console.error(`Failed to load documents: ${response.status}`);
        console.error(`Response text: ${await response.text()}`);
        // Try fallback endpoint
        return await loadDocumentsFromFallback();
      }
      
      const data = await response.json();
      
      if (data.documents && Array.isArray(data.documents)) {
        setDocuments(data.documents);
      } else {
        // If documents property is missing but we have data that looks like documents
        if (Array.isArray(data) && data.length > 0 && data[0].id) {
          setDocuments(data); // Use array directly
        } else {
          console.warn('Unexpected data format:', data);
          setDocuments([]);
        }
      }
    } catch (error) {
      console.error('Error loading documents:', error);
      showNotification(`Error loading documents: ${error.message}`, 'error');
      // Try fallback endpoint
      await loadDocumentsFromFallback();
    } finally {
      setIsLoading(false);
    }
  };
  
  // Fallback function for loading documents from alternative endpoint
  const loadDocumentsFromFallback = async () => {
    try {
      console.log('Trying fallback endpoint for documents');
      const response = await fetch(`${API_BASE_URL}/rag/simple-documents`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`Fallback request failed: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.documents && Array.isArray(data.documents)) {
        setDocuments(data.documents);
        showNotification('Documents loaded using fallback method', 'info');
        return true;
      } else {
        setDocuments([]);
        return false;
      }
    } catch (error) {
      console.error('Fallback document loading failed:', error);
      showNotification(`Could not load documents: ${error.message}`, 'error');
      setDocuments([]);
      return false;
    }
  };

  // Handle file selection
  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setDocumentTitle(file.name.split('.').slice(0, -1).join('.'));
  };

  // Handle file drop
  const handleFileDrop = (e) => {
    e.preventDefault();
    uploadAreaRef.current.classList.remove('bg-hover-color');
    
    if (e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  // Handle upload document
  const handleUploadDocument = async () => {
    if (!selectedFile) {
      showNotification('Please select a file to upload', 'error');
      return;
    }
    
    setIsUploading(true);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    if (documentTitle.trim()) {
      formData.append('title', documentTitle.trim());
    }
    
    if (documentTags.trim()) {
      formData.append('tags', documentTags.trim());
    }
    
    try {
      const response = await fetch(`${API_BASE_URL}/rag/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('Upload failed. Please try again.');
      }
      
      const data = await response.json();
      showNotification(`Document "${data.title}" uploaded successfully and is being processed.`, 'success');
      
      // Reset form
      setSelectedFile(null);
      setDocumentTitle('');
      setDocumentTags('');
      
      // Switch to documents tab after successful upload
      setTimeout(() => {
        setActiveTab('documents');
      }, 1500);
    } catch (error) {
      showNotification(error.message, 'error');
    } finally {
      setIsUploading(false);
    }
  };

  // Handle cancel upload
  const handleCancelUpload = () => {
    setSelectedFile(null);
    setDocumentTitle('');
    setDocumentTags('');
  };

  // Handle delete document
  const handleDeleteDocument = async (docId) => {
    if (!window.confirm('Are you sure you want to delete this document?')) {
      return;
    }
    
    try {
      const response = await fetch(`${API_BASE_URL}/rag/documents/${docId}`, {
        method: 'DELETE'
      });
      
      const data = await response.json();
      
      if (data.success) {
        showNotification('Document deleted successfully.', 'success');
        loadDocuments();
      } else {
        showNotification(`Error: ${data.error || 'Failed to delete document'}`, 'error');
      }
    } catch (error) {
      showNotification(`Error: ${error.message}`, 'error');
    }
  };

  // Handle search documents
  const handleSearchDocuments = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    setSearchResults([]);
    
    try {
      const response = await fetch(`${API_BASE_URL}/rag/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery.trim(),
          top_k: 5
        })
      });
      
      const data = await response.json();
      setSearchResults(data.results || []);
      
      if (!data.results || data.results.length === 0) {
        showNotification('No results found. Try a different search term.', 'info');
      }
    } catch (error) {
      showNotification(`Search error: ${error.message}`, 'error');
    } finally {
      setIsSearching(false);
    }
  };

  // Show notification
  const showNotification = (message, type) => {
    setNotification({
      message,
      type,
      visible: true
    });
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
      setNotification(prev => ({ ...prev, visible: false }));
    }, 5000);
  };

  return (
    <div className="flex h-screen">
      <Sidebar 
        isMobile={isMobile}
        showSidebar={showSidebar}
      />
      
      <div className="flex-1 flex flex-col h-screen max-h-screen overflow-hidden">
        <div className="p-4 border-b border-border-color flex items-center">
          <button 
            className={`p-1 text-muted-color hover:bg-hover-color rounded md:hidden`}
            onClick={() => setShowSidebar(!showSidebar)}
          >
            <Bars3Icon className="w-6 h-6" />
          </button>
          
          <h1 className="text-lg font-semibold ml-2">Document Management</h1>
        </div>
        
        {/* Notification */}
        {notification.visible && (
          <div className={`mx-6 mt-4 p-4 rounded-lg ${
            notification.type === 'success' ? 'bg-green-500/20 text-green-600' :
            notification.type === 'error' ? 'bg-red-500/20 text-red-600' : 
            'bg-blue-500/20 text-blue-600'
          }`}>
            {notification.message}
          </div>
        )}
        
        {/* Tabs */}
        <div className="flex border-b border-border-color mt-2">
          <button 
            className={`px-4 py-2 font-medium ${activeTab === 'upload' ? 'border-b-2 border-accent-color text-accent-color' : 'text-muted-color'}`}
            onClick={() => setActiveTab('upload')}
          >
            Upload
          </button>
          <button 
            className={`px-4 py-2 font-medium ${activeTab === 'documents' ? 'border-b-2 border-accent-color text-accent-color' : 'text-muted-color'}`}
            onClick={() => setActiveTab('documents')}
          >
            Documents
          </button>
          <button 
            className={`px-4 py-2 font-medium ${activeTab === 'search' ? 'border-b-2 border-accent-color text-accent-color' : 'text-muted-color'}`}
            onClick={() => setActiveTab('search')}
          >
            Search
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-6">
          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <div className="bg-card-bg rounded-lg shadow-md overflow-hidden">
              <div className="p-4 border-b border-border-color">
                <h2 className="text-lg font-medium">Upload Document</h2>
              </div>
              
              <div className="p-6">
                {!selectedFile ? (
                  <div 
                    ref={uploadAreaRef}
                    className="border-2 border-dashed border-accent-color/50 rounded-lg p-8 text-center cursor-pointer hover:bg-hover-color transition-colors"
                    onClick={() => fileInputRef.current.click()}
                    onDragOver={(e) => {
                      e.preventDefault();
                      uploadAreaRef.current.classList.add('bg-hover-color');
                    }}
                    onDragLeave={() => uploadAreaRef.current.classList.remove('bg-hover-color')}
                    onDrop={handleFileDrop}
                  >
                    <DocumentIcon className="w-10 h-10 mx-auto text-accent-color mb-3" />
                    <p className="mb-2">Drag & drop files here or click to select files</p>
                    <p className="text-sm text-muted-color">PDF, DOCX, TXT, CSV, and other text formats are supported</p>
                    <input 
                      type="file" 
                      ref={fileInputRef} 
                      className="hidden"
                      onChange={(e) => {
                        if (e.target.files.length > 0) {
                          handleFileSelect(e.target.files[0]);
                        }
                      }}
                    />
                  </div>
                ) : (
                  <div className="space-y-4">
                    {isUploading ? (
                      <div className="text-center py-8">
                        <div className="inline-block w-10 h-10 border-4 border-t-accent-color rounded-full animate-spin mb-4"></div>
                        <p className="text-lg font-medium mb-2">Uploading {selectedFile.name}...</p>
                        <p className="text-sm text-muted-color">Please wait while your document is being processed.</p>
                      </div>
                    ) : (
                      <>
                        <div className="flex items-center p-3 bg-hover-color rounded-lg">
                          <DocumentIcon className="w-6 h-6 text-accent-color mr-3" />
                          <div className="flex-1 overflow-hidden">
                            <p className="font-medium truncate">{selectedFile.name}</p>
                            <p className="text-sm text-muted-color">
                              {selectedFile.size < 1024 * 1024
                                ? `${(selectedFile.size / 1024).toFixed(1)} KB`
                                : `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB`}
                            </p>
                          </div>
                        </div>
                      
                        <div className="space-y-3">
                          <div>
                            <label className="block text-sm font-medium mb-1">Document Title</label>
                            <input
                              type="text"
                              value={documentTitle}
                              onChange={(e) => setDocumentTitle(e.target.value)}
                              className="w-full p-2 bg-input-bg border border-border-color rounded"
                              placeholder="Enter document title"
                            />
                          </div>
                          
                          <div>
                            <label className="block text-sm font-medium mb-1">Tags (comma separated)</label>
                            <input
                              type="text"
                              value={documentTags}
                              onChange={(e) => setDocumentTags(e.target.value)}
                              className="w-full p-2 bg-input-bg border border-border-color rounded"
                              placeholder="tag1, tag2, tag3"
                            />
                          </div>
                        </div>
                        
                        <div className="flex space-x-3 mt-4">
                          <button
                            className="px-4 py-2 bg-accent-color text-white rounded hover:bg-accent-hover transition-colors"
                            onClick={handleUploadDocument}
                          >
                            Upload Document
                          </button>
                          <button
                            className="px-4 py-2 bg-error-color/20 text-error-color rounded hover:bg-error-color/30 transition-colors"
                            onClick={handleCancelUpload}
                          >
                            Cancel
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Documents Tab */}
          {activeTab === 'documents' && (
            <div className="bg-card-bg rounded-lg shadow-md overflow-hidden">
              <div className="p-4 border-b border-border-color flex justify-between items-center">
                <h2 className="text-lg font-medium">Document List</h2>
                <button 
                  className="p-1.5 text-sm bg-hover-color text-muted-color rounded flex items-center gap-1.5 hover:text-text-color transition-colors"
                  onClick={loadDocuments}
                >
                  <ArrowPathIcon className="w-4 h-4" />
                  Refresh
                </button>
              </div>
              
              <div className="p-4">
                {isLoading ? (
                  <div className="text-center py-8">
                    <div className="inline-block w-8 h-8 border-4 border-t-accent-color rounded-full animate-spin mb-2"></div>
                    <p className="text-muted-color">Loading documents...</p>
                  </div>
                ) : documents.length === 0 ? (
                  <div className="text-center py-8">
                    <DocumentIcon className="w-10 h-10 mx-auto text-muted-color mb-3" />
                    <p className="text-muted-color">No documents found. Upload some documents to get started.</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {documents.map((doc) => {
                      // Format date
                      let dateAdded = 'Unknown date';
                      if (doc.date_added) {
                        const date = new Date(doc.date_added);
                        dateAdded = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
                      }
                      
                      return (
                        <div key={doc.id} className="border border-border-color rounded-lg p-4 hover:bg-hover-color/50 transition-colors">
                          <div className="flex justify-between">
                            <div>
                              <h3 className="font-medium text-lg">{doc.title || 'Untitled'}</h3>
                              <p className="text-sm text-muted-color">Filename: {doc.filename}</p>
                              <p className="text-sm text-muted-color">Added: {dateAdded}</p>
                              <p className="text-sm text-muted-color">Chunks: {doc.chunk_count || 0}</p>
                              
                              {doc.tags && doc.tags.length > 0 && (
                                <div className="flex flex-wrap gap-2 mt-2">
                                  {doc.tags.map((tag, index) => (
                                    <span key={index} className="px-2 py-1 bg-accent-color/20 text-accent-color rounded-full text-xs">
                                      {tag}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                            
                            <div className="flex space-x-2">
                              <button
                                className="p-2 text-muted-color hover:text-text-color hover:bg-hover-color rounded-full transition-colors"
                                onClick={() => window.open(`${API_BASE_URL}/rag/documents/${doc.id}/download`, '_blank')}
                                title="Download document"
                              >
                                <DocumentArrowDownIcon className="w-5 h-5" />
                              </button>
                              <button
                                className="p-2 text-error-color hover:bg-error-color/20 rounded-full transition-colors"
                                onClick={() => handleDeleteDocument(doc.id)}
                                title="Delete document"
                              >
                                <TrashIcon className="w-5 h-5" />
                              </button>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Search Tab */}
          {activeTab === 'search' && (
            <div className="bg-card-bg rounded-lg shadow-md overflow-hidden">
              <div className="p-4 border-b border-border-color">
                <h2 className="text-lg font-medium">Search Documents</h2>
              </div>
              
              <div className="p-6">
                <div className="flex space-x-2 mb-6">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearchDocuments()}
                    className="flex-1 p-2 bg-input-bg border border-border-color rounded"
                    placeholder="Enter search query"
                  />
                  <button
                    className="px-4 py-2 bg-accent-color text-white rounded hover:bg-accent-hover transition-colors"
                    onClick={handleSearchDocuments}
                  >
                    Search
                  </button>
                </div>
                
                {isSearching ? (
                  <div className="text-center py-8">
                    <div className="inline-block w-8 h-8 border-4 border-t-accent-color rounded-full animate-spin mb-2"></div>
                    <p className="text-muted-color">Searching documents...</p>
                  </div>
                ) : searchResults.length > 0 ? (
                  <div>
                    <p className="mb-4">Found {searchResults.length} results for "{searchQuery}"</p>
                    <div className="space-y-4">
                      {searchResults.map((result, index) => (
                        <div key={index} className="border-l-4 border-accent-color bg-hover-color/30 p-4 rounded-r-lg">
                          <h3 className="font-medium mb-2">{index + 1}. {result.title || 'Untitled'}</h3>
                          <p className="text-sm text-muted-color mb-2">
                            Score: {result.rerank_score ? result.rerank_score.toFixed(3) : result.score.toFixed(3)}
                          </p>
                          <div className="bg-input-bg p-3 rounded whitespace-pre-wrap text-sm">
                            {result.text}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : searchQuery ? (
                  <div className="text-center py-8 text-muted-color">
                    No results found. Try a different search term.
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-color">
                    Enter a search term to find content across your documents.
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
      
      <BriefingModal
        isOpen={briefingModal.isOpen}
        onClose={() => setBriefingModal({ ...briefingModal, isOpen: false })}
        briefingTitle={briefingModal.title}
        briefingFilename={briefingModal.filename}
      />
    </div>
  );
};

export default DocumentPage;