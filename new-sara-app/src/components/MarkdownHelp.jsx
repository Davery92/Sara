import React from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';

const MarkdownHelp = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div 
        className="bg-bg-color rounded-lg w-11/12 max-w-2xl max-h-[90vh] flex flex-col overflow-hidden shadow-lg animate-[modal-fade-in_0.3s_ease]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-4 border-b border-border-color flex items-center justify-between">
          <h3 className="text-lg font-semibold">Markdown Guide</h3>
          <button 
            className="text-muted-color p-1 rounded hover:text-text-color hover:bg-hover-color transition-colors"
            onClick={onClose}
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>
        
        <div className="p-6 overflow-y-auto">
          <h4 className="text-lg font-medium mb-4">Basic Syntax</h4>
          
          <div className="mb-6 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-bg p-3 rounded">
                <code className="text-sm font-mono"># Heading 1</code>
              </div>
              <div>
                <h1 className="text-2xl font-bold">Heading 1</h1>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-bg p-3 rounded">
                <code className="text-sm font-mono">## Heading 2</code>
              </div>
              <div>
                <h2 className="text-xl font-bold">Heading 2</h2>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-bg p-3 rounded">
                <code className="text-sm font-mono">### Heading 3</code>
              </div>
              <div>
                <h3 className="text-lg font-bold">Heading 3</h3>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-bg p-3 rounded">
                <code className="text-sm font-mono">**Bold text**</code>
              </div>
              <div>
                <strong>Bold text</strong>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-bg p-3 rounded">
                <code className="text-sm font-mono">*Italic text*</code>
              </div>
              <div>
                <em>Italic text</em>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-bg p-3 rounded">
                <code className="text-sm font-mono">[Link text](https://example.com)</code>
              </div>
              <div>
                <a href="#" className="text-accent-color underline">Link text</a>
              </div>
            </div>
          </div>
          
          <h4 className="text-lg font-medium mb-4">Lists</h4>
          
          <div className="mb-6 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-bg p-3 rounded">
                <code className="text-sm font-mono whitespace-pre-wrap">
                  {`- Item 1\n- Item 2\n- Item 3`}
                </code>
              </div>
              <div>
                <ul className="list-disc list-inside">
                  <li>Item 1</li>
                  <li>Item 2</li>
                  <li>Item 3</li>
                </ul>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-bg p-3 rounded">
                <code className="text-sm font-mono whitespace-pre-wrap">
                  {`1. First item\n2. Second item\n3. Third item`}
                </code>
              </div>
              <div>
                <ol className="list-decimal list-inside">
                  <li>First item</li>
                  <li>Second item</li>
                  <li>Third item</li>
                </ol>
              </div>
            </div>
          </div>
          
          <h4 className="text-lg font-medium mb-4">Other Elements</h4>
          
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-bg p-3 rounded">
                <code className="text-sm font-mono whitespace-pre-wrap">
                  {`> This is a blockquote`}
                </code>
              </div>
              <div>
                <blockquote className="pl-4 border-l-4 border-accent-color italic">
                  This is a blockquote
                </blockquote>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-bg p-3 rounded">
                <code className="text-sm font-mono whitespace-pre-wrap">
                  {`\`inline code\``}
                </code>
              </div>
              <div>
                <code className="bg-input-bg px-1 rounded text-sm font-mono">inline code</code>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-bg p-3 rounded">
                <code className="text-sm font-mono whitespace-pre-wrap">
                  {`\`\`\`\ncode block\n\`\`\``}
                </code>
              </div>
              <div>
                <pre className="bg-input-bg p-2 rounded text-sm font-mono overflow-x-auto">code block</pre>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-bg p-3 rounded">
                <code className="text-sm font-mono whitespace-pre-wrap">
                  {`---`}
                </code>
              </div>
              <div>
                <hr className="border-t border-border-color my-2" />
              </div>
            </div>
          </div>
        </div>
        
        <div className="p-4 border-t border-border-color">
          <p className="text-sm text-muted-color">
            This is a basic guide to Markdown syntax. For more details, see the <a href="https://www.markdownguide.org/" target="_blank" rel="noopener noreferrer" className="text-accent-color">Markdown Guide</a>.
          </p>
        </div>
      </div>
    </div>
  );
};

export default MarkdownHelp;