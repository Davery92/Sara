import React, { useState, useEffect } from 'react';
import { ArrowPathIcon, PencilIcon, PlusIcon, TrashIcon, CheckIcon, XMarkIcon } from '@heroicons/react/24/outline';

const CoreMemoriesCard = ({ onRefresh }) => {
  const [memories, setMemories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedMemories, setEditedMemories] = useState([]);
  const [newMemory, setNewMemory] = useState('');
  const [showAddForm, setShowAddForm] = useState(false);

  const loadMemories = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/v1/memory/core');
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      setMemories(data.memories || []);
    } catch (err) {
      setError(`Error loading core memories: ${err.message}`);
      console.error('Error fetching core memories:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMemories();
  }, []);

  const handleStartEdit = () => {
    setEditedMemories([...memories]);
    setIsEditing(true);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditedMemories([]);
  };

  const handleUpdateMemory = (index, value) => {
    const updated = [...editedMemories];
    updated[index] = value;
    setEditedMemories(updated);
  };

  const handleDeleteMemory = (index) => {
    const updated = [...editedMemories];
    updated.splice(index, 1);
    setEditedMemories(updated);
  };

  const handleSaveMemories = async () => {
    try {
      setLoading(true);
      
      const response = await fetch('/v1/memory/core/rewrite', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ memories: editedMemories.filter(m => m.trim() !== '') })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      // Reload memories
      await loadMemories();
      setIsEditing(false);
    } catch (err) {
      setError(`Error saving memories: ${err.message}`);
      console.error('Error saving memories:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddMemory = async () => {
    if (!newMemory.trim()) return;
    
    try {
      setLoading(true);
      
      const response = await fetch('/v1/memory/core', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ memory: newMemory.trim() })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      // Reload memories
      await loadMemories();
      setNewMemory('');
      setShowAddForm(false);
    } catch (err) {
      setError(`Error adding memory: ${err.message}`);
      console.error('Error adding memory:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-card-bg rounded-lg overflow-hidden">
      <div className="p-4 border-b border-border-color flex justify-between items-center">
        <h3 className="font-medium">Core Memories</h3>
        <div className="flex items-center gap-2">
          {!isEditing && (
            <>
              <button 
                className="p-1.5 text-sm bg-hover-color text-muted-color rounded flex items-center gap-1.5 hover:text-text-color transition-colors"
                onClick={() => setShowAddForm(true)}
              >
                <PlusIcon className="w-4 h-4" />
                Add
              </button>
              <button 
                className="p-1.5 text-sm bg-hover-color text-muted-color rounded flex items-center gap-1.5 hover:text-text-color transition-colors"
                onClick={handleStartEdit}
              >
                <PencilIcon className="w-4 h-4" />
                Edit
              </button>
            </>
          )}
          <button 
            className="p-1.5 text-sm bg-hover-color text-muted-color rounded flex items-center gap-1.5 hover:text-text-color transition-colors"
            onClick={loadMemories}
            disabled={loading}
          >
            <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>
      
      <div className="p-4">
        {error && (
          <div className="text-error-color mb-4 p-3 bg-error-color/10 rounded-md">
            {error}
          </div>
        )}
        
        {/* Add Memory Form */}
        {showAddForm && !isEditing && (
          <div className="mb-4 p-3 bg-input-bg rounded-md">
            <textarea 
              className="w-full bg-card-bg border border-border-color rounded p-2 mb-3"
              placeholder="Enter new memory..."
              value={newMemory}
              onChange={(e) => setNewMemory(e.target.value)}
              rows={3}
            />
            <div className="flex justify-end gap-2">
              <button 
                className="px-3 py-1.5 bg-hover-color text-text-color rounded"
                onClick={() => setShowAddForm(false)}
              >
                Cancel
              </button>
              <button 
                className="px-3 py-1.5 bg-accent-color text-white rounded"
                onClick={handleAddMemory}
                disabled={!newMemory.trim()}
              >
                Add Memory
              </button>
            </div>
          </div>
        )}
        
        {loading && !isEditing ? (
          <div className="text-center py-4">
            <div className="inline-block w-6 h-6 border-2 border-t-accent-color rounded-full animate-spin"></div>
            <p className="mt-2 text-muted-color">Loading core memories...</p>
          </div>
        ) : memories.length === 0 && !isEditing ? (
          <div className="text-center py-4 text-muted-color">
            No core memories available
          </div>
        ) : (
          <div>
            {!isEditing ? (
              <div className="space-y-2">
                {memories.map((memory, index) => (
                  <div key={index} className="p-3 bg-input-bg rounded-md">
                    {memory}
                  </div>
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                {editedMemories.map((memory, index) => (
                  <div key={index} className="flex gap-2">
                    <textarea 
                      className="flex-1 bg-input-bg border border-border-color rounded p-2"
                      value={memory}
                      onChange={(e) => handleUpdateMemory(index, e.target.value)}
                      rows={2}
                    />
                    <button 
                      className="p-1 text-error-color hover:bg-error-color/10 rounded h-fit"
                      onClick={() => handleDeleteMemory(index)}
                    >
                      <TrashIcon className="w-5 h-5" />
                    </button>
                  </div>
                ))}
                
                <div className="flex justify-end gap-2 mt-4">
                  <button 
                    className="px-3 py-1.5 bg-hover-color text-text-color rounded flex items-center gap-1"
                    onClick={handleCancelEdit}
                  >
                    <XMarkIcon className="w-4 h-4" />
                    Cancel
                  </button>
                  <button 
                    className="px-3 py-1.5 bg-accent-color text-white rounded flex items-center gap-1"
                    onClick={handleSaveMemories}
                  >
                    <CheckIcon className="w-4 h-4" />
                    Save Changes
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default CoreMemoriesCard;