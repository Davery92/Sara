import React, { useState, useEffect } from 'react';
import { CheckCircleIcon, XCircleIcon, InformationCircleIcon, ExclamationTriangleIcon, XMarkIcon } from '@heroicons/react/24/outline';

// Toast types with their respective icons and styles
const TOAST_TYPES = {
  success: {
    icon: CheckCircleIcon,
    className: 'bg-green-900/20 text-green-500 border-green-500/30'
  },
  error: {
    icon: XCircleIcon,
    className: 'bg-red-900/20 text-red-500 border-red-500/30'
  },
  info: {
    icon: InformationCircleIcon,
    className: 'bg-blue-900/20 text-blue-500 border-blue-500/30'
  },
  warning: {
    icon: ExclamationTriangleIcon,
    className: 'bg-yellow-900/20 text-yellow-500 border-yellow-500/30'
  }
};

// Individual Toast component
const Toast = ({ id, message, type = 'info', onClose, duration = 5000 }) => {
  const [visible, setVisible] = useState(true);
  
  // Get the icon and styles for this toast type
  const { icon: ToastIcon, className } = TOAST_TYPES[type] || TOAST_TYPES.info;

  useEffect(() => {
    // Auto-dismiss toast after duration
    const timer = setTimeout(() => {
      setVisible(false);
      setTimeout(() => onClose(id), 300); // Remove after fade animation
    }, duration);

    return () => clearTimeout(timer);
  }, [id, duration, onClose]);

  return (
    <div 
      className={`flex items-center p-4 mb-3 rounded-lg border shadow-lg transition-all duration-300 ${className} ${
        visible ? 'opacity-100 transform translate-y-0' : 'opacity-0 transform -translate-y-2'
      }`}
    >
      <ToastIcon className="w-5 h-5 mr-3 flex-shrink-0" />
      <p className="flex-1">{message}</p>
      <button 
        onClick={() => {
          setVisible(false);
          setTimeout(() => onClose(id), 300);
        }}
        className="p-1 ml-3 rounded-full hover:bg-white/10 transition-colors"
      >
        <XMarkIcon className="w-4 h-4" />
      </button>
    </div>
  );
};

// Toast container that manages all active toasts
const ToastContainer = () => {
  const [toasts, setToasts] = useState([]);

  // Add a new toast
  const addToast = (message, type = 'info', duration = 5000) => {
    const id = Date.now().toString();
    setToasts(prev => [...prev, { id, message, type, duration }]);
    return id;
  };

  // Remove a toast by ID
  const removeToast = (id) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };

  // Expose methods to the window object so they can be called from anywhere
  useEffect(() => {
    window.showToast = addToast;
    return () => {
      delete window.showToast;
    };
  }, []);

  return (
    <>
      <div className="fixed top-4 right-4 z-50 w-72 flex flex-col items-end">
        {toasts.map(toast => (
          <Toast
            key={toast.id}
            id={toast.id}
            message={toast.message}
            type={toast.type}
            duration={toast.duration}
            onClose={removeToast}
          />
        ))}
      </div>
    </>
  );
};

export default ToastContainer;