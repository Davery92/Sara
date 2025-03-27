import React, { useState } from 'react';

const NotesList = ({ notes = [], onViewAll }) => {
  const [viewingNote, setViewingNote] = useState(null);

  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch (e) {
      return dateString || 'Unknown date';
    }
  };

  const handleViewNote = (note) => {
    setViewingNote(note);
  };

  const closeNoteModal = () => {
    setViewingNote(null);
  };

  return (
    <div className="bg-card-bg rounded-lg overflow-hidden">
      <div className="p-4 border-b border-border-color flex justify-between items-center">
        <h3 className="font-medium">Recent Notes</h3>
        <button 
          className="px-3 py-1.5 text-sm bg-accent-color hover:bg-accent-hover text-white rounded"
          onClick={onViewAll}
        >
          View All
        </button>
      </div>
      
      <div className="overflow-x-auto">
        {notes.length === 0 ? (
          <div className="p-4 text-center text-muted-color">
            No notes available
          </div>
        ) : (
          <table className="w-full">
            <thead className="bg-hover-color/50">
              <tr>
                <th className="py-3 px-4 text-left text-sm font-medium">Title</th>
                <th className="py-3 px-4 text-left text-sm font-medium">Created</th>
                <th className="py-3 px-4 text-left text-sm font-medium">Last Modified</th>
                <th className="py-3 px-4 text-left text-sm font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {notes.map((note, index) => (
                <tr key={note.filename || index} className="border-b border-border-color">
                  <td className="py-3 px-4">{note.title}</td>
                  <td className="py-3 px-4 text-sm">{formatDate(note.created)}</td>
                  <td className="py-3 px-4 text-sm">{formatDate(note.last_modified)}</td>
                  <td className="py-3 px-4">
                    <button 
                      className="px-3 py-1.5 text-xs bg-hover-color hover:bg-card-bg text-text-color rounded"
                      onClick={() => handleViewNote(note)}
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      
      {/* Note Modal */}
      {viewingNote && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50"
          onClick={closeNoteModal}
        >
          <div 
            className="bg-bg-color rounded-lg w-11/12 max-w-3xl max-h-[90vh] flex flex-col overflow-hidden shadow-lg"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-4 border-b border-border-color flex justify-between items-center">
              <h3 className="text-lg font-semibold">{viewingNote.title}</h3>
              <button 
                className="text-muted-color p-1 rounded hover:text-text-color hover:bg-hover-color"
                onClick={closeNoteModal}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
            
            <div className="p-6 overflow-y-auto">
              <div className="mb-4">
                <p className="text-sm text-muted-color">Created: {formatDate(viewingNote.created)}</p>
                <p className="text-sm text-muted-color">Last Modified: {formatDate(viewingNote.last_modified)}</p>
                
                {viewingNote.tags && viewingNote.tags.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-2">
                    {viewingNote.tags.map((tag, i) => (
                      <span key={i} className="px-2 py-1 bg-accent-color/20 text-accent-color rounded-full text-xs">
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              
              <div className="p-4 bg-input-bg rounded whitespace-pre-wrap">
                {viewingNote.content}
              </div>
            </div>
            
            <div className="p-4 border-t border-border-color flex justify-end">
              <button 
                className="px-4 py-2 bg-hover-color hover:bg-card-bg text-text-color rounded"
                onClick={closeNoteModal}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NotesList;