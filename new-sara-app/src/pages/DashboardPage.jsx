import { useState, useEffect } from 'react';
import { Bars3Icon } from '@heroicons/react/24/outline';
import Sidebar from '../components/Sidebar';
import BriefingModal from '../components/BriefingModal';
import StatCard from '../components/StatCard';
import SystemStatusCard from '../components/SystemStatusCard';
import NotesList from '../components/NotesList';
import BriefingsList from '../components/BriefingsList';
import CoreMemoriesCard from '../components/CoreMemoriesCard';

const DashboardPage = () => {
  const [showSidebar, setShowSidebar] = useState(true);
  const [isMobile, setIsMobile] = useState(false);
  const [stats, setStats] = useState({
    notesCount: 0,
    coreMemoriesCount: 0,
    activeModel: '',
    gpuMemoryUsage: '',
    tokenUsage: {
      value: 0,
      remaining: 0,
      percentage: 0
    }
  });
  const [systemStatus, setSystemStatus] = useState({
    server: 'Checking...',
    redis: 'Checking...',
    neo4j: 'Checking...'
  });
  const [notes, setNotes] = useState([]);
  const [briefings, setBriefings] = useState([]);
  const [briefingModal, setBriefingModal] = useState({
    isOpen: false,
    title: '',
    filename: ''
  });
  const [loading, setLoading] = useState(true);

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

  // Load dashboard data on mount
  useEffect(() => {
    const loadDashboardData = async () => {
      setLoading(true);
      
      try {
        // Load data concurrently
        await Promise.all([
          loadSystemStatus(),
          loadGpuStats(),
          loadNotes(),
          loadBriefings(),
          loadTokenUsage()
        ]);
      } catch (error) {
        console.error('Error loading dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();
  }, []);

  // Make global briefing modal function available
  useEffect(() => {
    window.openBriefingModal = (filename, briefings) => {
      const briefing = briefings.find(b => b.filename === filename) || { title: filename };
      setBriefingModal({
        isOpen: true,
        title: briefing.title,
        filename: filename
      });
    };

    return () => {
      delete window.openBriefingModal;
    };
  }, []);

  const loadSystemStatus = async () => {
    try {
      const response = await fetch('/health');
      const data = await response.json();
      
      setSystemStatus({
        server: data.services?.server === 'online' ? 'Online' : 'Offline',
        redis: data.services?.redis === 'connected' ? 'Connected' : 'Disconnected',
        neo4j: data.services?.neo4j || 'Unknown'
      });
    } catch (error) {
      console.error('Error loading system status:', error);
      // Use fallback values
      setSystemStatus({
        server: 'Unknown',
        redis: 'Unknown',
        neo4j: 'Unknown'
      });
    }
  };

  const loadGpuStats = async () => {
    try {
      const response = await fetch('/v1/system/gpu');
      const data = await response.json();
      
      // Update stats
      setStats(prevStats => ({
        ...prevStats,
        activeModel: 'qwen2.5:32b', // Default model or from another API
        gpuMemoryUsage: `${data.memory_used?.toFixed(1) || 0} GB / ${data.memory_total?.toFixed(1) || 0} GB`
      }));
    } catch (error) {
      console.error('Error loading GPU stats:', error);
    }
  };

  const loadNotes = async () => {
    try {
      const notesResp = await fetch('/v1/notes');
      const notesData = await notesResp.json();
      
      // Get core memories
      const memoriesResp = await fetch('/v1/memory/core');
      const memoriesData = await memoriesResp.json();
      
      // Update stats
      setStats(prevStats => ({
        ...prevStats,
        notesCount: notesData.notes?.length || 0,
        coreMemoriesCount: memoriesData.memories?.length || 0
      }));
      
      // Store notes for display
      setNotes(notesData.notes?.slice(0, 5) || []);
    } catch (error) {
      console.error('Error loading notes:', error);
    }
  };

  const loadBriefings = async () => {
    try {
      const response = await fetch('/v1/briefings/list');
      const data = await response.json();
      
      if (data.error) {
        console.error('Error loading briefings:', data.error);
      } else {
        setBriefings(data.briefings || []);
      }
    } catch (error) {
      console.error('Error fetching briefings:', error);
    }
  };

  const loadTokenUsage = async () => {
    try {
      const response = await fetch('/v1/stats/tokens');
      const data = await response.json();
      
      // Calculate token usage percentage
      const totalTokens = data.total_tokens || 0;
      const remaining = data.total_tokens ? (32768 - data.total_tokens) : 50000;
      const percentage = Math.min(Math.round((totalTokens / 32768) * 100), 100);
      
      setStats(prevStats => ({
        ...prevStats,
        tokenUsage: {
          value: totalTokens,
          remaining: remaining,
          percentage: percentage
        }
      }));
    } catch (error) {
      console.error('Error loading token usage:', error);
      // Use fallback values
      setStats(prevStats => ({
        ...prevStats,
        tokenUsage: {
          value: 0,
          remaining: 32768,
          percentage: 0
        }
      }));
    }
  };

  const handleOpenBriefing = (filename) => {
    const briefing = briefings.find(b => b.filename === filename) || { title: filename };
    setBriefingModal({
      isOpen: true,
      title: briefing.title,
      filename: filename
    });
  };

  return (
    <div className="flex h-screen">
      <Sidebar 
        isMobile={isMobile}
        showSidebar={showSidebar}
      />
      
      <div className="flex-1 flex flex-col h-screen max-h-screen overflow-hidden">
        <div className="flex-1 overflow-y-auto">
          <div className="p-4 border-b border-border-color flex items-center">
            <button 
              className={`p-1 text-muted-color hover:bg-hover-color rounded md:hidden`}
              onClick={() => setShowSidebar(!showSidebar)}
            >
              <Bars3Icon className="w-6 h-6" />
            </button>
          
            <div className="flex flex-1 justify-end items-center gap-3">
              <div className={`px-3 py-1.5 rounded-full flex items-center gap-1.5 ${systemStatus.server === 'Online' ? 'bg-green-900/20 text-green-500' : 'bg-red-900/20 text-red-500'}`}>
                <span className={`w-2 h-2 rounded-full ${systemStatus.server === 'Online' ? 'bg-green-500' : 'bg-red-500'}`}></span>
                Server: {systemStatus.server}
              </div>
              
              <div className={`px-3 py-1.5 rounded-full flex items-center gap-1.5 ${systemStatus.redis === 'Connected' ? 'bg-green-900/20 text-green-500' : 'bg-yellow-900/20 text-yellow-500'}`}>
                <span className={`w-2 h-2 rounded-full ${systemStatus.redis === 'Connected' ? 'bg-green-500' : 'bg-yellow-500'}`}></span>
                Database: {systemStatus.redis}
              </div>
            </div>
          </div>
          
          <div className="p-6">
            <h2 className="text-xl font-semibold mb-6">Dashboard Overview</h2>
            
            {loading ? (
              <div className="flex justify-center items-center p-12">
                <div className="inline-block w-8 h-8 border-4 border-t-accent-color rounded-full animate-spin"></div>
                <span className="ml-3 text-muted-color">Loading dashboard data...</span>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                  <StatCard 
                    title="Total Notes"
                    value={stats.notesCount.toString()}
                    detail={`Last updated ${new Date().toLocaleTimeString()}`}
                  />
                  
                  <StatCard 
                    title="Core Memories"
                    value={stats.coreMemoriesCount.toString()}
                    detail={`Last updated ${new Date().toLocaleTimeString()}`}
                  />
                  
                  <StatCard 
                    title="Active Model"
                    value={stats.activeModel}
                    detail="Default Model"
                  />
                  
                  <StatCard 
                    title="GPU Memory Usage"
                    value={stats.gpuMemoryUsage}
                    detail="VRAM Utilization"
                  />
                  
                  <StatCard 
                    title="Token Usage"
                    value={stats.tokenUsage.value.toLocaleString()}
                    meter={{
                      percentage: stats.tokenUsage.percentage,
                      color: stats.tokenUsage.percentage > 80 ? 'red' : stats.tokenUsage.percentage > 60 ? 'yellow' : 'green'
                    }}
                    detail={`${stats.tokenUsage.remaining.toLocaleString()} remaining`}
                  />
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                  <SystemStatusCard 
                    systemStatus={systemStatus}
                    onRefresh={loadSystemStatus}
                  />
                  
                  <BriefingsList 
                    briefings={briefings}
                    onOpenBriefing={handleOpenBriefing}
                    onRefresh={loadBriefings}
                  />
                </div>
                
                {/* Core Memories Section */}
                <div className="mb-6">
                  <CoreMemoriesCard onRefresh={loadNotes} />
                </div>
                
                <div className="mt-6">
                  <NotesList 
                    notes={notes}
                    onViewAll={() => {/* Navigate to notes page */}}
                  />
                </div>
              </>
            )}
          </div>
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

export default DashboardPage;