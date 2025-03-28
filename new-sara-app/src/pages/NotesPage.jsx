import { useState, useEffect, useRef } from 'react';
import { 
  Bars3Icon, 
  DocumentPlusIcon, 
  FolderIcon, 
  ArrowUturnLeftIcon, 
  ArrowUturnRightIcon,
  QuestionMarkCircleIcon
} from '@heroicons/react/24/outline';
import Sidebar from '../components/Sidebar';
import ReactMarkdown from 'react-markdown';
import MarkdownHelp from '../components/MarkdownHelp';

const NotesPage = () => {
  const [showSidebar, setShowSidebar] = useState(true);
  const [isMobile, setIsMobile] = useState(false);
  const [notes, setNotes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedNote, setSelectedNote] = useState(null);
  const [editorContent, setEditorContent] = useState('');
  const [isPreviewMode, setIsPreviewMode] = useState(false);
  const [showNewNoteModal, setShowNewNoteModal] = useState(false);
  const [newNoteTitle, setNewNoteTitle] = useState('');
  const [newNoteTags, setNewNoteTags] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [editHistory, setEditHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [showMarkdownHelp, setShowMarkdownHelp] = useState(false);
  
  const editorRef = useRef(null);
  const saveTimeoutRef = useRef(null);

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

  // Load notes on mount
  useEffect(() => {
    loadNotes();
  }, []);

  // Add to history when editor content changes
  useEffect(() => {
    if (editorContent && historyIndex === editHistory.length - 1) {
      // Add to history if content changed and we're at the current end of history
      setEditHistory(prev => [...prev.slice(0, historyIndex + 1), editorContent]);
      setHistoryIndex(prev => prev + 1);
    } else if (editorContent && historyIndex !== editHistory.length - 1) {
      // If we're in the middle of history and content changes, truncate history and add new state
      setEditHistory(prev => [...prev.slice(0, historyIndex + 1), editorContent]);
      setHistoryIndex(prev => prev + 1);
    }
  }, [editorContent]);

  // Auto-save effect
  useEffect(() => {
    if (selectedNote && editorContent !== selectedNote.content) {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
      
      saveTimeoutRef.current = setTimeout(() => {
        saveNote();
      }, 2000); // Auto-save after 2 seconds of inactivity
    }
    
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [editorContent, selectedNote]);

  const loadNotes = async () => {
    setLoading(true);
    try {
      const response = await fetch('/v1/notes');
      const data = await response.json();
      
      if (data.notes) {
        setNotes(data.notes);
        
        // If no note is selected and there are notes, select the first one
        if (!selectedNote && data.notes.length > 0) {
          handleSelectNote(data.notes[0]);
        }
      }
    } catch (error) {
      console.error('Error loading notes:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectNote = (note) => {
    // Save current note if it exists and has changed
    if (selectedNote && editorContent !== selectedNote.content) {
      saveNote();
    }
    
    setSelectedNote(note);
    setEditorContent(note.content || '');
    // Reset history
    setEditHistory([note.content || '']);
    setHistoryIndex(0);
  };

  const createNewNote = async () => {
    if (!newNoteTitle.trim()) {
      alert('Please enter a title for the new note');
      return;
    }
    
    setIsSaving(true);
    try {
      const tags = newNoteTags.split(',')
        .map(tag => tag.trim())
        .filter(tag => tag.length > 0);
      
      const response = await fetch('/v1/notes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          title: newNoteTitle,
          content: '',
          tags: tags
        })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Error creating note');
      }
      
      // Reload notes and select the new one
      await loadNotes();
      const createdNote = notes.find(note => note.title === newNoteTitle);
      if (createdNote) {
        handleSelectNote(createdNote);
      }
      
      // Reset new note form
      setNewNoteTitle('');
      setNewNoteTags('');
      setShowNewNoteModal(false);
    } catch (error) {
      console.error('Error creating note:', error);
      alert(`Failed to create note: ${error.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const saveNote = async () => {
    if (!selectedNote) return;
    
    setIsSaving(true);
    try {
      const response = await fetch(`/v1/notes/${selectedNote.filename}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          content: editorContent
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error saving note');
      }
      
      // Update the note in the local state
      setSelectedNote(prev => ({
        ...prev,
        content: editorContent,
        last_modified: new Date().toISOString()
      }));
      
      // Update the note in the notes list
      setNotes(prev => prev.map(note => 
        note.filename === selectedNote.filename 
          ? {...note, content: editorContent, last_modified: new Date().toISOString()} 
          : note
      ));
    } catch (error) {
      console.error('Error saving note:', error);
      alert(`Failed to save note: ${error.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const deleteNote = async () => {
    if (!selectedNote) return;
    
    if (!window.confirm(`Are you sure you want to delete "${selectedNote.title}"?`)) {
      return;
    }
    
    try {
      const response = await fetch(`/v1/notes/${selectedNote.filename}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error deleting note');
      }
      
      // Remove the note from the local state
      setNotes(prev => prev.filter(note => note.filename !== selectedNote.filename));
      
      // If there are still notes, select the first one, otherwise clear the selection
      if (notes.length > 1) {
        const nextNote = notes.find(note => note.filename !== selectedNote.filename);
        if (nextNote) {
          handleSelectNote(nextNote);
        }
      } else {
        setSelectedNote(null);
        setEditorContent('');
      }
    } catch (error) {
      console.error('Error deleting note:', error);
      alert(`Failed to delete note: ${error.message}`);
    }
  };

  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch (e) {
      return dateString || 'Unknown date';
    }
  };

  // History navigation functions
  const handleUndo = () => {
    if (historyIndex > 0) {
      setHistoryIndex(prev => prev - 1);
      setEditorContent(editHistory[historyIndex - 1]);
    }
  };

  const handleRedo = () => {
    if (historyIndex < editHistory.length - 1) {
      setHistoryIndex(prev => prev + 1);
      setEditorContent(editHistory[historyIndex + 1]);
    }
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
        
          <div className="ml-4">
            <h2 className="text-lg font-semibold">Notes</h2>
          </div>

          <div className="flex flex-1 justify-end items-center gap-3">
            <button
              className="p-1.5 text-sm bg-hover-color text-muted-color rounded flex items-center gap-1.5 hover:text-text-color transition-colors"
              onClick={() => setIsPreviewMode(!isPreviewMode)}
            >
              {isPreviewMode ? 'Edit Mode' : 'Preview Mode'}
            </button>
            
            <button
              className="p-1.5 text-sm bg-hover-color text-muted-color rounded flex items-center gap-1.5 hover:text-text-color transition-colors"
              onClick={() => setShowMarkdownHelp(true)}
            >
              <QuestionMarkCircleIcon className="w-4 h-4" />
              Markdown Help
            </button>

            <button
              className="p-1.5 text-sm bg-accent-color text-white rounded flex items-center gap-1.5 hover:bg-accent-hover transition-colors"
              onClick={() => setShowNewNoteModal(true)}
            >
              <DocumentPlusIcon className="w-4 h-4" />
              New Note
            </button>
          </div>
        </div>
        
        <div className="flex-1 flex overflow-hidden">
          {/* Notes sidebar */}
          <div className="w-64 border-r border-border-color overflow-y-auto bg-card-bg">
            <div className="p-3 border-b border-border-color">
              <h3 className="font-medium">All Notes</h3>
            </div>
            
            {loading ? (
              <div className="p-4 text-center text-muted-color">
                <div className="inline-block w-5 h-5 border-2 border-t-accent-color rounded-full animate-spin mb-2"></div>
                <p>Loading notes...</p>
              </div>
            ) : notes.length === 0 ? (
              <div className="p-4 text-center text-muted-color">
                <FolderIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No notes yet</p>
                <button
                  className="mt-2 px-3 py-1.5 text-xs bg-accent-color text-white rounded"
                  onClick={() => setShowNewNoteModal(true)}
                >
                  Create your first note
                </button>
              </div>
            ) : (
              <div className="divide-y divide-border-color">
                {notes.map((note) => (
                  <div
                    key={note.filename}
                    className={`p-3 cursor-pointer transition-colors hover:bg-hover-color ${
                      selectedNote && selectedNote.filename === note.filename ? 'bg-hover-color' : ''
                    }`}
                    onClick={() => handleSelectNote(note)}
                  >
                    <h4 className="font-medium">{note.title}</h4>
                    <p className="text-xs text-muted-color">
                      {formatDate(note.last_modified || note.created)}
                    </p>
                    {note.tags && note.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        {note.tags.map((tag, i) => (
                          <span key={i} className="px-1.5 py-0.5 bg-accent-color/20 text-accent-color rounded-sm text-xs">
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
          
          {/* Editor/Preview */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {selectedNote ? (
              <>
                <div className="p-2 border-b border-border-color bg-input-bg flex items-center">
                  <div className="flex-1">
                    <h3 className="font-medium">{selectedNote.title}</h3>
                    <p className="text-xs text-muted-color">
                      {formatDate(selectedNote.last_modified || selectedNote.created)}
                    </p>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <button
                      className="p-1.5 text-muted-color hover:text-text-color disabled:opacity-50"
                      onClick={handleUndo}
                      disabled={historyIndex <= 0}
                    >
                      <ArrowUturnLeftIcon className="w-4 h-4" />
                    </button>
                    <button
                      className="p-1.5 text-muted-color hover:text-text-color disabled:opacity-50"
                      onClick={handleRedo}
                      disabled={historyIndex >= editHistory.length - 1}
                    >
                      <ArrowUturnRightIcon className="w-4 h-4" />
                    </button>
                    <button
                      className="p-1.5 text-sm bg-hover-color text-muted-color rounded hover:text-text-color"
                      onClick={saveNote}
                      disabled={isSaving}
                    >
                      {isSaving ? 'Saving...' : 'Save'}
                    </button>
                    <button
                      className="p-1.5 text-sm bg-hover-color text-error-color rounded hover:bg-error-color hover:text-white"
                      onClick={deleteNote}
                    >
                      Delete
                    </button>
                  </div>
                </div>
                
                <div className="flex-1 overflow-auto">
                  {isPreviewMode ? (
                    <div className="p-6 prose prose-invert max-w-none">
                      <ReactMarkdown>
                        {editorContent}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <textarea
                      ref={editorRef}
                      className="w-full h-full p-6 bg-bg-color text-text-color resize-none font-mono text-sm leading-relaxed focus:outline-none"
                      value={editorContent}
                      onChange={(e) => setEditorContent(e.target.value)}
                      placeholder="Type your markdown here..."
                    />
                  )}
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center p-6 text-center">
                <div>
                  <DocumentPlusIcon className="w-16 h-16 mx-auto mb-4 text-muted-color" />
                  <h3 className="text-xl font-medium mb-2">No Note Selected</h3>
                  <p className="text-muted-color mb-4">Select a note from the sidebar or create a new one</p>
                  <button
                    className="px-4 py-2 bg-accent-color text-white rounded"
                    onClick={() => setShowNewNoteModal(true)}
                  >
                    Create New Note
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* New Note Modal */}
      {showNewNoteModal && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50"
          onClick={() => setShowNewNoteModal(false)}
        >
          <div 
            className="bg-bg-color rounded-lg w-96 max-w-full overflow-hidden shadow-lg animate-[modal-fade-in_0.3s_ease]"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-4 border-b border-border-color">
              <h3 className="text-lg font-semibold">Create New Note</h3>
            </div>
            
            <div className="p-6">
              <div className="mb-4">
                <label className="block text-sm font-medium mb-1">Title</label>
                <input
                  type="text"
                  className="w-full p-2 bg-input-bg border border-border-color rounded text-text-color"
                  value={newNoteTitle}
                  onChange={(e) => setNewNoteTitle(e.target.value)}
                  placeholder="Note title"
                />
              </div>
              
              <div className="mb-6">
                <label className="block text-sm font-medium mb-1">Tags (comma separated)</label>
                <input
                  type="text"
                  className="w-full p-2 bg-input-bg border border-border-color rounded text-text-color"
                  value={newNoteTags}
                  onChange={(e) => setNewNoteTags(e.target.value)}
                  placeholder="tag1, tag2, tag3"
                />
              </div>
              
              <div className="flex justify-end gap-3">
                <button
                  className="px-4 py-2 bg-hover-color text-text-color rounded"
                  onClick={() => setShowNewNoteModal(false)}
                >
                  Cancel
                </button>
                <button
                  className="px-4 py-2 bg-accent-color text-white rounded"
                  onClick={createNewNote}
                  disabled={isSaving}
                >
                  {isSaving ? 'Creating...' : 'Create Note'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NotesPage;