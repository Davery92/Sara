import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import ChatPage from './pages/ChatPage';
import DashboardPage from './pages/DashboardPage';
import NotesPage from './pages/NotesPage';
import DocumentPage from './pages/DocumentPage';
import './index.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/notes" element={<NotesPage />} />
        <Route path="/documents" element={<DocumentPage />} />
      </Routes>
    </Router>
  );
}

export default App;